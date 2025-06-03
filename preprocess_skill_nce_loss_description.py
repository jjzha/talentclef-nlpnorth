import os
import random
import logging
import pandas as pd
import ast
from datasets import Dataset

# ----------------------------
# Setup logging
# ----------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

def sample_negatives(current_positive, pool, num_negatives):
    """Samples `num_negatives` items from `pool` that ≠ current_positive."""
    negatives = []
    while len(negatives) < num_negatives:
        neg = random.choice(pool)
        if neg != current_positive:
            negatives.append(neg)
    return negatives

# --- PART A: Original ESCO‐based pipeline --- #

def create_nce_from_esco(job2skills_df, occ_df, skills_df,
                         num_negatives=5, seed=42):
    """
    From your job2skill.tsv + occupations_en.csv + skills_en.csv,
    build NCE samples exactly as before (job_uri→skill_uri).
    """
    random.seed(seed)
    # build jobs dict
    jobs = {}
    for _, row in occ_df.iterrows():
        uri  = str(row["conceptUri"]).strip()
        pref = str(row["preferredLabel"]).strip()
        alts = []
        if "altLabels" in row and pd.notna(row["altLabels"]):
            alts = [s.strip() for s in str(row["altLabels"]).split("\n") if s.strip()]
        desc = str(row["description"]).strip() if "description" in row and pd.notna(row["description"]) else ""
        jobs[uri] = {"pref": pref, "alt": alts, "desc": desc}

    # build skills dict
    skills = {}
    for _, row in skills_df.iterrows():
        uri  = str(row["conceptUri"]).strip()
        pref = str(row["preferredLabel"]).strip()
        alts = []
        if "altLabels" in row and pd.notna(row["altLabels"]):
            alts = [s.strip() for s in str(row["altLabels"]).split("\n") if s.strip()]
        desc = str(row["description"]).strip() if "description" in row and pd.notna(row["description"]) else ""
        skills[uri] = {"pref": pref, "alt": alts, "desc": desc}

    # build negative pools
    pool_sk = []
    for s in skills.values():
        base = f"{s['pref']} {s['desc']}".strip()
        pool_sk.append(base)
        for a in s['alt']:
            if a != s['pref']:
                pool_sk.append(f"{a} {s['desc']}".strip())
    pool_jk = []
    for j in jobs.values():
        base = f"{j['pref']} {j['desc']}".strip()
        pool_jk.append(base)
        for a in j['alt']:
            if a != j['pref']:
                pool_jk.append(f"{a} {j['desc']}".strip())

    samples = []
    for _, row in job2skills_df.iterrows():
        job_uri   = str(row["job_id"]).strip()
        skill_uri = str(row["skill_id"]).strip()
        if job_uri not in jobs or skill_uri not in skills:
            continue

        j = jobs[job_uri]
        s = skills[skill_uri]
        job_text = f"{j['pref']} {j['desc']}".strip()
        pos_text = f"{s['pref']} {s['desc']}".strip()

        # Sample 1
        negs = sample_negatives(pos_text, pool_sk, num_negatives)
        samples.append({
            "anchor": job_text,
            "positive": pos_text,
            "negatives": negs,
            "pair_type": "job_skill"
        })

        # Sample 2: skill alts
        for alt in s['alt']:
            if alt == s['pref']: continue
            alt_text = f"{alt} {s['desc']}".strip()
            negs = sample_negatives(alt_text, pool_sk, num_negatives)
            samples.append({
                "anchor": job_text,
                "positive": alt_text,
                "negatives": negs,
                "pair_type": "job_skill_alt"
            })

        # Sample 3: job alts
        for alt in j['alt']:
            if alt == j['pref']: continue
            alt_job = f"{alt} {j['desc']}".strip()
            negs = sample_negatives(alt_job, pool_jk, num_negatives)
            samples.append({
                "anchor": alt_job,
                "positive": pos_text,
                "negatives": negs,
                "pair_type": "skill_job_alt"
            })

    logging.info(f"[ESCO] Generated {len(samples)} samples.")
    return samples

# --- PART B: English folder skill‐alias pipeline --- #

def build_job_skill_mapping_from_folder(folder):
    """
    Reads queries.tsv, qrels.tsv and corpus_elements.tsv in `folder`,
    parses & explodes skill_aliases, and returns a DataFrame with
      job_title | skill_alias
    """
    queries = pd.read_csv(os.path.join(folder, "queries"), sep="\t")
    qrels   = pd.read_csv(os.path.join(folder, "qrels.tsv"),
                         sep="\t", header=None,
                         names=["q_id","ignore","c_id","rel"])
    corpus  = pd.read_csv(os.path.join(folder, "corpus_elements"), sep="\t")

    # parse list-of-aliases
    corpus["skill_aliases"] = corpus["skill_aliases"].apply(ast.literal_eval)

    merged = (
        qrels
          .merge(queries, on="q_id")    # brings jobtitle
          .merge(corpus, on="c_id")     # brings skill_aliases
    )
    exploded = merged.explode("skill_aliases")
    df = exploded[["jobtitle","skill_aliases"]].rename(
         columns={"jobtitle":"job_title","skill_aliases":"skill"}
    )
    logging.info(f"[Folder] Built {len(df)} job→skill_alias rows in '{folder}'")
    return df

def create_nce_from_aliases(mapping_df, num_negatives=5, seed=42):
    """
    From a mapping of (job_title, skill) strings,
    generate only simple job_skill NCE samples.
    """
    random.seed(seed)
    pool = mapping_df["skill"].tolist()
    samples = []
    for _, row in mapping_df.iterrows():
        anchor   = row["job_title"]
        positive = row["skill"]
        negs     = sample_negatives(positive, pool, num_negatives)
        samples.append({
            "anchor": anchor,
            "positive": positive,
            "negatives": negs,
            "pair_type": "job_skill_alias"
        })
    logging.info(f"[Alias] Generated {len(samples)} samples.")
    return samples

# --- PART C: Orchestration & saving --- #

def preprocess_and_save_nce(
    job2skills_path,
    occ_csv_path,
    skills_csv_path,
    english_folder="data/TaskB/validation/english",
    output_path="data/combined_nce",
    num_negatives=16,
    seed=42,
    test_size=0.001
):
    # Load original ESCO inputs
    logging.info("Loading ESCO job2skills + CSVs…")
    job2skills_df = pd.read_csv(job2skills_path, sep="\t", header=None,
                                names=["job_id","skill_id","relation"])
    occ_df        = pd.read_csv(occ_csv_path)
    skills_df     = pd.read_csv(skills_csv_path)

    # A) ESCO‐based samples
    samples_esco = create_nce_from_esco(
        job2skills_df, occ_df, skills_df,
        num_negatives=num_negatives, seed=seed
    )

    # B) English‐folder‐based skill‐aliases
    mapping_alias = build_job_skill_mapping_from_folder(english_folder)
    samples_alias = create_nce_from_aliases(
        mapping_alias, num_negatives=num_negatives, seed=seed
    )

    # Merge both
    all_samples = samples_esco + samples_alias
    logging.info(f"Total combined samples: {len(all_samples)}")

    # To HuggingFace Dataset
    ds = Dataset.from_dict({
        "anchor":    [s["anchor"]    for s in all_samples],
        "positive":  [s["positive"]  for s in all_samples],
        "negatives": [s["negatives"] for s in all_samples],
        "pair_type": [s["pair_type"] for s in all_samples],
    })

    # train/test split
    ds = ds.train_test_split(test_size=test_size, seed=seed)
    logging.info(f"Train size: {len(ds['train'])}, Test size: {len(ds['test'])}")

    # save
    ds.save_to_disk(output_path)
    logging.info(f"Combined NCE dataset saved to {output_path}")

if __name__ == "__main__":
    preprocess_and_save_nce(
        job2skills_path="data/TaskB/training/job2skill.tsv",
        occ_csv_path="data/occupations_en.csv",
        skills_csv_path="data/skills_en.csv",
        english_folder="data/TaskB/validation/english",
        output_path="data/TaskB/train_data_nce_skill_16neg_description_final",
        num_negatives=16,
        seed=42,
        test_size=0.001
    )
