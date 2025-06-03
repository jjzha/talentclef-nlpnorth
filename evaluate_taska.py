import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from ranx import Qrels, Run, evaluate
import argparse

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)

#### TALENTCLEF SCRIPT ####

def load_qrels(qrels_path):
    """
    Loads the qrels file (TREC format: q_id, iter, doc_id, rel)
    and converts it to a Qrels object.
    """
    qrels_df = pd.read_csv(qrels_path, sep="\t", header=None,
                           names=["q_id", "iter", "doc_id", "rel"],
                           dtype={"q_id": str, "doc_id": str, "rel": int})
    return Qrels.from_df(qrels_df, q_id_col="q_id", doc_id_col="doc_id", score_col="rel")

def load_run(run_path):
    """
    Loads the run file (TREC format: q_id, Q0, doc_id, rank, score, [tag])
    and converts it to a Run object.
    """
    with open(run_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    header_present = "q_id" in first_line.lower() or "q0" in first_line.lower()
    if header_present:
        run_df = pd.read_csv(run_path, sep=r"\s+", header=0)
    else:
        run_df = pd.read_csv(run_path, sep=r"\s+", header=None)

    if run_df.shape[1] == 5:
        run_df.columns = ["q_id", "Q0", "doc_id", "rank", "score"]
    elif run_df.shape[1] >= 6:
        run_df.columns = ["q_id", "Q0", "doc_id", "rank", "score", "tag"]
    else:
        raise ValueError("The run file does not have the expected format.")

    run_df["q_id"] = run_df.q_id.astype(str)
    run_df["doc_id"] = run_df.doc_id.astype(str)
    run_df["score"] = run_df["score"].astype(float)

    return Run.from_df(run_df, q_id_col="q_id", doc_id_col="doc_id", score_col="score")

def run_evaluation(qrels_path, output_file):
    print("Received parameters:")
    print(f"  qrels: {qrels_path}")
    print(f"  run: {output_file}")

    print("Loading qrels...")
    qrels = load_qrels(qrels_path)
    print("Loading run...")
    run = load_run(output_file)

    metrics = ["map", "mrr", "ndcg", "precision@5", "precision@10", "precision@100"]

    print("Running evaluation...")
    results = evaluate(qrels, run, metrics, make_comparable=True)

    results_file = os.path.join(os.path.dirname(output_file), "evaluation.results")
    with open(results_file, "a") as f:
        f.write(f"\n=== Evaluation Results {output_file} ===\n")
        for metric, score in results.items():
            f.write(f"{metric}: {score:.4f}\n")

    return results

#### TALENTCLEF END ####

def load_queries_corpus(queries_path, corpus_path):
    """
    Load queries and corpus elements from TSV files, skipping the header.
    """
    logging.info(f"Loading queries from {queries_path}")
    queries_df = pd.read_csv(queries_path, sep="\t", names=["q_id", "jobtitle"], skiprows=1)
    logging.info(f"Loading corpus from {corpus_path}")
    corpus_df = pd.read_csv(corpus_path, sep="\t", names=["c_id", "jobtitle"], skiprows=1)
    return queries_df, corpus_df

def compute_similarities(model, queries_df, corpus_df, min_top_k, max_top_k, score_threshold,
                         cross_encoder_model=None, rerank_top_k=None, instruct_prompt=None, e5_prefix=False):
    """
    Compute cosine similarity between queries and corpus elements.
    Optionally re-rank with a cross-encoder.
    Returns a list of TREC-formatted result strings.
    """
    # Always retrieve all corpus documents
    max_top_k = len(corpus_df)

    # Encode
    if instruct_prompt:
        logging.info("Using generative retrieval mode with prompt...")
        query_embeddings = model.encode(
            queries_df["jobtitle"].tolist(),
            prompt=instruct_prompt,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        corpus_embeddings = model.encode(corpus_df["jobtitle"].tolist(), convert_to_tensor=True)
    elif e5_prefix:
        logging.info("Using E5-style prefixes...")
        query_texts = ["query: " + q for q in queries_df["jobtitle"]]
        corpus_texts = ["passage: " + c for c in corpus_df["jobtitle"]]
        query_embeddings = model.encode(query_texts, convert_to_tensor=True)
        corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True)
    else:
        logging.info("Encoding without prefixes...")
        query_embeddings = model.encode(queries_df["jobtitle"].tolist(), convert_to_tensor=True)
        corpus_embeddings = model.encode(corpus_df["jobtitle"].tolist(), convert_to_tensor=True)

    similarities = util.cos_sim(query_embeddings, corpus_embeddings).cpu().numpy()
    corpus_map = dict(zip(corpus_df["c_id"].astype(str), corpus_df["jobtitle"]))

    results = []
    for q_idx, q_id in enumerate(queries_df["q_id"]):
        query_text = queries_df.iloc[q_idx]["jobtitle"]
        sorted_idxs = np.argsort(-similarities[q_idx])
        rank = 1
        added = 0
        candidates = []
        for c_idx in sorted_idxs:
            score = similarities[q_idx, c_idx]
            if score_threshold is not None and score < score_threshold and added >= min_top_k:
                break
            doc_id = corpus_df.iloc[c_idx]["c_id"]
            candidates.append({
                "doc_id": doc_id,
                "baseline_score": score,
                "baseline_rank": rank
            })
            rank += 1
            added += 1
            if added >= max_top_k:
                break

        # optional cross-encoder
        if cross_encoder_model and candidates:
            top = candidates[:rerank_top_k] if rerank_top_k else candidates
            pairs = [(query_text, corpus_map[str(c["doc_id"])]) for c in top]
            logging.info(f"Re-ranking {len(pairs)} for query {q_id}...")
            cross_scores = cross_encoder_model.predict(pairs)
            for i, c in enumerate(top):
                c["cross_score"] = cross_scores[i]
            top = sorted(top, key=lambda x: x["cross_score"], reverse=True)
            for new_rank, c in enumerate(top, start=1):
                results.append(f"{int(q_id)} Q0 {int(c['doc_id'])} {new_rank} {c['cross_score']:.4f} cross_encoder")
        else:
            for c in candidates:
                results.append(f"{int(q_id)} Q0 {int(c['doc_id'])} {c['baseline_rank']} {c['baseline_score']:.4f} baseline_model")

    return results

def save_predictions(results, output_path, tag):
    """
    Save results in TREC Run format, with a uniform tag.
    """
    logging.info(f"Saving predictions to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("q_id\tQ0\tdoc_id\trank\tscore\ttag\n")
        for line in results:
            qid, Q0, docid, rank, score, *rest = line.split()
            f.write(f"{qid}\t{Q0}\t{docid}\t{rank}\t{score}\t{tag}\n")

def evaluate_thresholds(validation_root, model, lang, min_top_k, max_top_k, thresholds, output_dir,
                        cross_encoder_model=None, rerank_top_k=None, instruct_prompt=None, e5_prefix=False):
    """
    Full eval over thresholds for a single language.
    """
    queries_path = os.path.join(validation_root, lang, "queries")
    corpus_path = os.path.join(validation_root, lang, "corpus_elements")
    qrels_path = os.path.join(validation_root, lang, "qrels.tsv")

    if not (os.path.exists(queries_path) and os.path.exists(corpus_path) and os.path.exists(qrels_path)):
        logging.warning(f"Skipping {lang}, missing data.")
        return

    qdf, cdf = load_queries_corpus(queries_path, corpus_path)
    scores = []
    for thr in thresholds:
        out_run = os.path.join(output_dir, f"eval_{lang}_{thr:.2f}.trec")
        res = compute_similarities(
            model, qdf, cdf,
            min_top_k, max_top_k, thr,
            cross_encoder_model, rerank_top_k, instruct_prompt, e5_prefix
        )
        save_predictions(res, out_run, tag="baseline_model")
        er = run_evaluation(qrels_path, out_run)
        scores.append(er.get("map", 0))
    plt.figure()
    plt.plot(thresholds, scores, marker='o')
    plt.xlabel("Threshold")
    plt.ylabel("MAP")
    plt.title(f"{lang} Threshold vs MAP")
    plt.grid()
    plt.savefig(os.path.join(output_dir, f"thr_curve_{lang}.png"))
    plt.close()

def evaluate_all_languages(validation_root, model_path, output_dir, languages, min_top_k, max_top_k, thresholds,
                           cross_encoder_model_path=None, rerank_top_k=None, instruct_prompt=None, e5_prefix=False):
    """
    Full evaluation across multiple languages.
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Loading SentenceTransformer from {model_path}")
    model = SentenceTransformer(model_path, trust_remote_code=True)

    cross_encoder = None
    if cross_encoder_model_path:
        logging.info(f"Loading CrossEncoder from {cross_encoder_model_path}")
        cross_encoder = CrossEncoder(cross_encoder_model_path, trust_remote_code=True)

    for lang in languages:
        logging.info(f"=== Evaluating {lang} ===")
        evaluate_thresholds(
            validation_root, model, lang,
            min_top_k, max_top_k, thresholds, output_dir,
            cross_encoder, rerank_top_k, instruct_prompt, e5_prefix
        )

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate or generate (cross- or mono-lingual) predictions with SentenceTransformers."
    )
    parser.add_argument("--validation_root", type=str, default="data/TaskA/validation")
    parser.add_argument("--model_path",    type=str, default="models/mpnet/checkpoint")
    parser.add_argument("--output_dir",    type=str, default="evals/")
    parser.add_argument("--languages",     type=str, default="chinese,english,german,spanish",
                        help="Comma-separated for mono-lingual eval.")
    parser.add_argument("--cross_lingual", type=str, default=None,
                        help="Comma-separated source-target pairs, e.g. en-de,en-es,en-zh")
    parser.add_argument("--min_top_k",     type=int, default=1)
    parser.add_argument("--max_top_k",     type=int, default=100)
    parser.add_argument("--threshold_start", type=float, default=0.05)
    parser.add_argument("--threshold_end",   type=float, default=0.95)
    parser.add_argument("--threshold_step",  type=float, default=0.05)
    parser.add_argument("--cross_encoder_model", type=str, default=None)
    parser.add_argument("--rerank_top_k",       type=int, default=None)
    parser.add_argument("--instruct_prompt",     type=str, default="")
    parser.add_argument("--e5_prefix",           action="store_true")
    parser.add_argument("--generate_only",       action="store_true",
                        help="Skip qrels eval; just dump predictions.")
    parser.add_argument("--custom_tag",          type=str, default="baseline_model")

    args = parser.parse_args()
    instr = args.instruct_prompt.strip() or None

    if args.generate_only:
        os.makedirs(args.output_dir, exist_ok=True)
        logging.info("Loading model for prediction-only mode...")
        model = SentenceTransformer(args.model_path, trust_remote_code=True)
        cross_encoder = None
        if args.cross_encoder_model:
            logging.info("Loading cross-encoder for prediction-only mode...")
            cross_encoder = CrossEncoder(args.cross_encoder_model, trust_remote_code=True)

        # cross-lingual or mono-lingual?
        if args.cross_lingual:
            pairs = [p.split("-") for p in args.cross_lingual.split(",")]
            for src, tgt in pairs:
                qp = os.path.join(args.validation_root, src, "queries")
                cp = os.path.join(args.validation_root, tgt, "corpus_elements")
                qdf, cdf = load_queries_corpus(qp, cp)
                res = compute_similarities(
                    model, qdf, cdf,
                    args.min_top_k, args.max_top_k, None,
                    cross_encoder, args.rerank_top_k, instr, args.e5_prefix
                )
                out = os.path.join(args.output_dir, f"predictions_{src}-{tgt}.trec")
                save_predictions(res, out, args.custom_tag)
                logging.info(f"Saved cross-lingual {src}->{tgt} to {out}")
        else:
            langs = args.languages.split(",")
            for lang in langs:
                qp = os.path.join(args.validation_root, lang, "queries")
                cp = os.path.join(args.validation_root, lang, "corpus_elements")
                qdf, cdf = load_queries_corpus(qp, cp)
                res = compute_similarities(
                    model, qdf, cdf,
                    args.min_top_k, args.max_top_k, None,
                    cross_encoder, args.rerank_top_k, instr, args.e5_prefix
                )
                out = os.path.join(args.output_dir, f"predictions_{lang}.trec")
                save_predictions(res, out, args.custom_tag)
                logging.info(f"Saved mono-lingual {lang} to {out}")

        logging.info("Prediction-only mode complete.")

    else:
        # full evaluation mode
        thresholds = np.arange(args.threshold_start, args.threshold_end, args.threshold_step)
        langs = args.languages.split(",")
        evaluate_all_languages(
            validation_root=args.validation_root,
            model_path=args.model_path,
            output_dir=args.output_dir,
            languages=langs,
            min_top_k=args.min_top_k,
            max_top_k=args.max_top_k,
            thresholds=thresholds,
            cross_encoder_model_path=args.cross_encoder_model,
            rerank_top_k=args.rerank_top_k,
            instruct_prompt=instr,
            e5_prefix=args.e5_prefix
        )

if __name__ == "__main__":
    main()
