#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Revised Evaluation Script for TalentCLEF Task B using Internal Evaluation (MAP only)
with Corpus Skills Extraction, Prediction-Only Mode, Custom Tags, and Cross-Lingual Support

This script assumes that each validation folder contains:
  - queries.tsv         (with header; columns: q_id, jobtitle)
  - corpus_elements.tsv (with header; columns: c_id, esco_uri, skill_aliases)
  - qrels.tsv           (TREC format without header)

Features added:
  * --generate_only       : skip evaluation, only output predictions
  * --custom_tag <tag>    : tag to assign to every row in prediction-only runs
  * --cross_lingual <pairs>: comma-separated src-tgt language codes (e.g. en-de,en-es,en-zh)
"""

import os
import sys
import logging
import argparse
import ast
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer, util
from ranx import Qrels, Run, evaluate

# Optionally import CodeCarbon for emissions tracking
try:
    from codecarbon import EmissionsTracker
except ImportError:
    EmissionsTracker = None

# ------------------------------------------------------------------------------
# Internal Evaluation Functions (using Ranx)
# ------------------------------------------------------------------------------

def load_qrels(qrels_path):
    qrels_df = pd.read_csv(qrels_path, sep="\t", header=None,
                           names=["q_id", "iter", "doc_id", "rel"],
                           dtype={"q_id": str, "doc_id": str, "rel": int})
    return Qrels.from_df(qrels_df, q_id_col="q_id", doc_id_col="doc_id", score_col="rel")

def load_run(run_path):
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

def run_evaluation(qrels_path, run_path):
    print(f"  evaluating run: {run_path}")
    qrels = load_qrels(qrels_path)
    run   = load_run(run_path)
    results = evaluate(qrels, run, ["map"], make_comparable=True)
    if not hasattr(results, "items"):
        results = {"map": float(results)}
    results_file = os.path.join(os.path.dirname(run_path), "evaluation.results")
    with open(results_file, "a") as f:
        f.write(f"\n=== Evaluation Results for {run_path} ===\n")
        for metric, score in results.items():
            f.write(f"{metric}: {score:.4f}\n")
    return results

# ------------------------------------------------------------------------------
# Data Loading (Queries and Corpus)
# ------------------------------------------------------------------------------

def load_queries_corpus(queries_tsv, corpus_tsv):
    logging.info(f"Loading queries from {queries_tsv}")
    queries_df = pd.read_csv(queries_tsv, sep="\t", skiprows=0, dtype=str)
    logging.info(f"Loading corpus from {corpus_tsv}")
    corpus_df  = pd.read_csv(corpus_tsv, sep="\t", skiprows=0, dtype=str)
    # parse skill_aliases list
    corpus_df["skill_aliases"] = corpus_df["skill_aliases"].apply(ast.literal_eval)
    # keep only first alias as primary_skill
    corpus_df["primary_skill"] = corpus_df["skill_aliases"].apply(
        lambda lst: lst[0] if isinstance(lst, list) and lst else ""
    )
    return queries_df, corpus_df

# ------------------------------------------------------------------------------
# Similarity Computation and Run File Creation
# ------------------------------------------------------------------------------

def compute_similarities(model, queries_df, corpus_df,
                         min_top_k, max_top_k, score_threshold,
                         cross_encoder_model=None, rerank_top_k=None,
                         instruct_prompt=None, e5_prefix=False):
    query_texts  = queries_df["jobtitle"].tolist()
    corpus_texts = corpus_df["primary_skill"].tolist()

    if instruct_prompt:
        logging.info("Encoding with instruct_prompt...")
        query_emb = model.encode(query_texts, prompt=instruct_prompt,
                                 convert_to_tensor=True, normalize_embeddings=True)
        corpus_emb = model.encode(corpus_texts, convert_to_tensor=True)
    elif e5_prefix:
        logging.info("Encoding with E5 prefixes...")
        query_emb  = model.encode([f"query: {q}" for q in query_texts], convert_to_tensor=True)
        corpus_emb = model.encode([f"passage: {c}" for c in corpus_texts], convert_to_tensor=True)
    else:
        logging.info("Encoding without prefixes...")
        query_emb  = model.encode(query_texts, convert_to_tensor=True)
        corpus_emb = model.encode(corpus_texts, convert_to_tensor=True)

    sims = util.cos_sim(query_emb, corpus_emb).cpu().numpy()
    results = []
    for qi, qid in enumerate(queries_df["q_id"]):
        ranked = np.argsort(-sims[qi])
        added = 0
        candidates = []
        rank = 1
        for ci in ranked:
            score = sims[qi, ci]
            if score_threshold is not None and score < score_threshold and added >= min_top_k:
                break
            doc_id = str(corpus_df.iloc[ci]["c_id"])
            if any(c["doc_id"] == doc_id for c in candidates):
                continue
            candidates.append({
                "doc_id": doc_id,
                "baseline_score": score,
                "baseline_rank": rank
            })
            rank += 1
            added += 1
            if added >= max_top_k:
                break

        if cross_encoder_model and candidates:
            topk = candidates[:rerank_top_k] if rerank_top_k else candidates
            pairs = [(queries_df.iloc[qi]["jobtitle"],
                      corpus_df[corpus_df["c_id"] == c["doc_id"]].iloc[0]["primary_skill"])
                     for c in topk]
            logging.info(f"Re-ranking {len(pairs)} for query {qid}")
            cross_scores = cross_encoder_model.predict(pairs)
            for i, c in enumerate(topk):
                c["cross_score"] = cross_scores[i]
            topk = sorted(topk, key=lambda x: x["cross_score"], reverse=True)
            for new_rank, c in enumerate(topk, start=1):
                results.append(f"{qid}\tQ0\t{c['doc_id']}\t{new_rank}\t{c['cross_score']:.4f}\tcross_encoder")
        else:
            for c in candidates:
                results.append(f"{qid}\tQ0\t{c['doc_id']}\t{c['baseline_rank']}\t{c['baseline_score']:.4f}\tbaseline_model")
    return results

def save_predictions(results, output_path, tag=None):
    logging.info(f"Saving run to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        if tag is not None:
            f.write("q_id\tQ0\tdoc_id\trank\tscore\ttag\n")
            for line in results:
                qid, Q0, docid, rank, score, *_ = line.split()
                f.write(f"{qid}\t{Q0}\t{docid}\t{rank}\t{score}\t{tag}\n")
        else:
            f.write("\n".join(results))

# ------------------------------------------------------------------------------
# Threshold-based Evaluation
# ------------------------------------------------------------------------------

def evaluate_thresholds(queries_tsv, corpus_tsv, qrels_tsv,
                        model, min_top_k, max_top_k, thresholds,
                        output_dir,
                        cross_encoder_model=None, rerank_top_k=None,
                        instruct_prompt=None, e5_prefix=False):
    queries_df, corpus_df = load_queries_corpus(queries_tsv, corpus_tsv)
    map_scores = []
    for thr in thresholds:
        out_run = os.path.join(output_dir, f"eval_{thr:.2f}.trec")
        logging.info(f"Threshold {thr:.2f}")
        res = compute_similarities(
            model, queries_df, corpus_df,
            min_top_k, max_top_k, thr,
            cross_encoder_model, rerank_top_k,
            instruct_prompt, e5_prefix
        )
        save_predictions(res, out_run)
        ev = run_evaluation(qrels_tsv, out_run)
        map_scores.append(ev.get("map", 0.0))
        print(f"  MAP @ {thr:.2f} = {map_scores[-1]:.4f}")

    plt.figure()
    plt.plot(thresholds, map_scores, marker='o', linestyle='-')
    plt.xlabel("Score Threshold")
    plt.ylabel("MAP")
    plt.title("Threshold vs MAP")
    plt.grid()
    curve_path = os.path.join(output_dir, "threshold_curve.png")
    plt.savefig(curve_path)
    plt.close()
    logging.info(f"Saved threshold curve to {curve_path}")

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TalentCLEF Task B evaluation & prediction script"
    )
    parser.add_argument("--validation_root",  type=str, required=True,
                        help="Folder with queries.tsv, corpus_elements.tsv, qrels.tsv")
    parser.add_argument("--model_path",       type=str, required=True,
                        help="Path to SentenceTransformer model")
    parser.add_argument("--output_dir",       type=str, default="evals",
                        help="Where to save runs and plots")
    parser.add_argument("--min_top_k",        type=int, default=1)
    parser.add_argument("--max_top_k",        type=int, default=100)
    parser.add_argument("--threshold_start",  type=float, default=0.05)
    parser.add_argument("--threshold_end",    type=float, default=0.95)
    parser.add_argument("--threshold_step",   type=float, default=0.05)
    parser.add_argument("--cross_encoder_model", type=str, default=None)
    parser.add_argument("--rerank_top_k",        type=int, default=None)
    parser.add_argument("--instruct_prompt",      type=str, default="")
    parser.add_argument("--e5_prefix",            action="store_true")
    parser.add_argument("--track_emissions",      action="store_true")
    # new args:
    parser.add_argument("--generate_only",        action="store_true",
                        help="Skip evaluation; only generate prediction runs")
    parser.add_argument("--custom_tag",           type=str, default="baseline_model",
                        help="Tag for every row in prediction-only mode")
    parser.add_argument("--cross_lingual",        type=str, default=None,
                        help="Comma-separated src-tgt pairs, e.g. en-de,en-es,en-zh")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

    # file paths for mono-lingual
    queries_tsv = os.path.join(args.validation_root, "queries")
    corpus_tsv  = os.path.join(args.validation_root, "corpus_elements")
    qrels_tsv   = os.path.join(args.validation_root, "qrels.tsv")

    # thresholds array
    thresholds = np.arange(args.threshold_start, args.threshold_end, args.threshold_step)

    # load models
    logging.info(f"Loading SentenceTransformer from {args.model_path}")
    model = SentenceTransformer(args.model_path, trust_remote_code=True)
    cross_encoder = None
    if args.cross_encoder_model:
        from sentence_transformers import CrossEncoder
        logging.info(f"Loading CrossEncoder from {args.cross_encoder_model}")
        cross_encoder = CrossEncoder(args.cross_encoder_model, trust_remote_code=True)

    # emissions tracking
    if args.track_emissions and EmissionsTracker:
        tracker = EmissionsTracker()
        tracker.start_task("evaluation")

    instruct = args.instruct_prompt.strip() or None

    if args.generate_only:
        # ------ prediction-only mode ------
        if args.cross_lingual:
            pairs = [p.split("-") for p in args.cross_lingual.split(",")]
            for src, tgt in pairs:
                q_src = os.path.join(args.validation_root, src, "queries")
                c_tgt = os.path.join(args.validation_root, tgt, "corpus_elements")
                qdf, cdf = load_queries_corpus(q_src, c_tgt)
                res = compute_similarities(
                    model, qdf, cdf,
                    args.min_top_k, args.max_top_k, None,
                    cross_encoder, args.rerank_top_k,
                    instruct, args.e5_prefix
                )
                out = os.path.join(args.output_dir, f"predictions_{src}-{tgt}.trec")
                save_predictions(res, out, tag=args.custom_tag)
                logging.info(f"Saved cross-lingual predictions {src}->{tgt} to {out}")
        else:
            # mono-lingual prediction
            qdf, cdf = load_queries_corpus(queries_tsv, corpus_tsv)
            res = compute_similarities(
                model, qdf, cdf,
                args.min_top_k, args.max_top_k, None,
                cross_encoder, args.rerank_top_k,
                instruct, args.e5_prefix
            )
            out = os.path.join(args.output_dir, "predictions.trec")
            save_predictions(res, out, tag=args.custom_tag)
            logging.info(f"Saved mono-lingual predictions to {out}")
    else:
        # ------ full evaluation mode ------
        evaluate_thresholds(
            queries_tsv, corpus_tsv, qrels_tsv,
            model, args.min_top_k, args.max_top_k, thresholds,
            args.output_dir,
            cross_encoder, args.rerank_top_k,
            instruct, args.e5_prefix
        )

    if args.track_emissions and EmissionsTracker:
        emissions = tracker.stop_task("evaluation")
        with open(os.path.join(args.output_dir, "emissions.json"), "w", encoding="utf-8") as f:
            json.dump(emissions, f, indent=2)

    logging.info("Done.")

if __name__ == "__main__":
    main()
