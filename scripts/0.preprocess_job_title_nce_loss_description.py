import os
import random
import logging
import pandas as pd
from datasets import Dataset

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

def sample_negatives(current_positive, pool, num_negatives):
    """Samples negatives from a pool ensuring they differ from the current positive."""
    negatives = []
    while len(negatives) < num_negatives:
        neg = random.choice(pool)
        if neg != current_positive:
            negatives.append(neg)
    return negatives

def create_nce_dataset(file_paths, num_negatives=5, seed=42):
    """
    Reads CSVs (with preferredLabel/description[/altLabels]) and TSVs 
    (where the last two columns are anchor→positive) and builds NCE samples.
    """
    random.seed(seed)
    logging.info(f"Processing files: {file_paths}")

    all_descriptions = []
    all_labels = []
    loaded = []  # tuples of (path, ext, df, lang)

    # ── First pass: load data + build negative pools ───────────────────────────────
    for path in file_paths:
        if not os.path.exists(path):
            logging.error(f"File not found: {path}")
            raise FileNotFoundError(f"{path} not found")

        ext = os.path.splitext(path)[1].lower()
        # derive 'language' tag from filename suffix:
        lang = os.path.basename(path).split("_")[-1].split(".")[0]

        if ext == ".csv":
            df = pd.read_csv(path)
            df = df.dropna(subset=["preferredLabel", "description"])
            # collect pools
            all_descriptions += df["description"].astype(str).tolist()
            all_labels       += df["preferredLabel"].astype(str).tolist()
            # include any altLabels in the pool too
            if "altLabels" in df.columns:
                for raw in df["altLabels"].fillna(""):
                    for alt in str(raw).split("\n"):
                        alt = alt.strip()
                        if alt:
                            all_labels.append(alt)
            loaded.append((path, "csv", df, lang))

        elif ext == ".tsv":
            df = pd.read_csv(path, sep="\t")
            # assume last two columns are anchor→positive
            a_col, p_col = df.columns[-2], df.columns[-1]
            df = df.dropna(subset=[a_col, p_col])
            # add both to label-pool
            all_labels += df[a_col].astype(str).tolist()
            all_labels += df[p_col].astype(str).tolist()
            loaded.append((path, "tsv", df, lang))

        else:
            logging.warning(f"Skipping unsupported file type: {path}")

    # ── Second pass: emit samples ────────────────────────────────────────────────
    samples = []
    for path, ftype, df, lang in loaded:
        if ftype == "csv":
            for _, row in df.iterrows():
                anchor = str(row["preferredLabel"]).strip()
                desc   = str(row["description"]).strip()

                # Sample 1: (anchor→description)
                negs = sample_negatives(desc, all_descriptions, num_negatives)
                samples.append({
                    "anchor":    anchor,
                    "positive":  desc,
                    "negatives": negs,
                    "language":  lang
                })

                # Sample 2+: each altLabel → negatives
                for raw in [row.get("altLabels", "")] if "altLabels" in df.columns else []:
                    for alt in str(raw).split("\n"):
                        alt = alt.strip()
                        if alt and alt != anchor:
                            negs = sample_negatives(alt, all_labels, num_negatives)
                            samples.append({
                                "anchor":    anchor,
                                "positive":  alt,
                                "negatives": negs,
                                "language":  lang
                            })

        else:  # ftype == "tsv"
            a_col, p_col = df.columns[-2], df.columns[-1]
            for _, row in df.iterrows():
                anchor   = str(row[a_col]).strip()
                positive = str(row[p_col]).strip()
                negs = sample_negatives(positive, all_labels, num_negatives)
                samples.append({
                    "anchor":    anchor,
                    "positive":  positive,
                    "negatives": negs,
                    "language":  lang
                })

    logging.info(f"Created {len(samples)} samples for NCELoss training.")
    return samples

def preprocess_and_save_nce(file_paths,
                            output_path="data/processed_nce_dataset",
                            num_negatives=16, seed=42):
    """
    Prepares an NCELoss-compatible dataset from mixed CSV/TSV files and saves it.
    """
    logging.info("Starting NCELoss data preprocessing...")
    samples = create_nce_dataset(file_paths,
                                 num_negatives=num_negatives,
                                 seed=seed)

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict({
        "anchor":    [s["anchor"]    for s in samples],
        "positive":  [s["positive"]  for s in samples],
        "negatives": [s["negatives"] for s in samples],
        "language":  [s["language"]  for s in samples],
    })

    # Split out a tiny eval set
    dataset = dataset.train_test_split(test_size=0.01, seed=seed)
    logging.info(f"Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")

    # Persist
    dataset.save_to_disk(output_path)
    logging.info(f"NCELoss-compatible dataset saved to {output_path}")

if __name__ == "__main__":
    files = [
        "data/occupations_en.csv",
        "data/occupations_de.csv",
        "data/occupations_es.csv",
        "data/job_titles_development.tsv",    # ← your human-labeled TSV
    ]
    preprocess_and_save_nce(files,
                            output_path="data/final_data_nce_16neg")
