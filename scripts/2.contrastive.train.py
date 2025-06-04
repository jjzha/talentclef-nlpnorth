import os
import sys
import logging
import argparse
from datasets import load_from_disk, interleave_datasets
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import ContrastiveLoss, MultipleNegativesRankingLoss, CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction, TripletEvaluator
from sentence_transformers.training_args import BatchSamplers

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)

def add_prefixes(example):
    # If the dataset uses triplets
    if "anchor" in example:
        example["anchor"] = ["query: " + anc for anc in example["anchor"]]
        example["positive"] = ["passage: " + pos for pos in example["positive"]]
        # Assuming negatives is a list of strings
        example["negatives"] = [["passage: " + neg for neg in l] for l in example["negatives"]]
    # Alternatively, if you use an evaluator with two text columns:
    elif "text1" in example and "text2" in example:
        example["text1"] = "query: " + example["text1"]
        example["text2"] = "passage: " + example["text2"]
    return example


def main(data_path, data_paths, model_name, output_dir, num_train_epochs, batch_size, learning_rate):
    logging.info("Starting training pipeline...")

    # Load preprocessed dataset(s)
    if data_paths and len(data_paths) > 0:
        logging.info("Loading and interleaving multiple datasets...")
        train_datasets = []
        test_datasets = []
        for path in data_paths:
            if not os.path.exists(path):
                logging.error(f"Dataset not found at {path}")
                raise FileNotFoundError(f"Dataset not found at {path}")
            ds = load_from_disk(path)
            train_ds = ds["train"]#.remove_columns(["language"])
            test_ds = ds["test"]#.remove_columns(["language"])
            train_datasets.append(train_ds)
            test_datasets.append(test_ds)
        if len(train_datasets) == 1:
            train_dataset = train_datasets[0]
            test_dataset = test_datasets[0]
        else:
            train_dataset = interleave_datasets(train_datasets, stopping_strategy="first_exhausted")
            test_dataset = interleave_datasets(test_datasets, stopping_strategy="first_exhausted")
        logging.info(f"Interleaved train size: {len(train_dataset)}, test size: {len(test_dataset)}")
    else:
        # Fallback: load a single dataset from data_path
        if not os.path.exists(data_path):
            logging.error(f"Processed dataset not found at {data_path}")
            raise FileNotFoundError(f"Dataset not found at {data_path}")
        dataset = load_from_disk(data_path)
        train_dataset = dataset["train"]#.remove_columns(["language"])
        test_dataset = dataset["test"]#.remove_columns(["language"])
        logging.info(f"Loaded train size: {len(train_dataset)}, test size: {len(test_dataset)}")

    if args.e5_prefix:
        logging.info("Adding prefixes to dataset for E5 model training...")
        train_dataset = train_dataset.map(add_prefixes, batched=True)
        test_dataset = test_dataset.map(add_prefixes, batched=True)

    # Load pre-trained model
    logging.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # Define loss function
    # loss = ContrastiveLoss(model)
    # loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=48)
    loss = MultipleNegativesRankingLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        # batch_sampler=BatchSamplers.NO_DUPLICATES,
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=10000,
        # eval_steps=500,
        save_strategy="best",
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        logging_steps=1,
    )

    evaluator = TripletEvaluator(
        anchors=test_dataset["anchor"],
        positives=test_dataset["positive"],
        negatives=test_dataset["negatives"],
        # name="occupations_test_eval",
        name="skill_test_eval",
    )

    # Create Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    # Start training
    logging.info("Starting training...")
    trainer.train()
    logging.info("Training completed.")

    # Evaluate model
    logging.info("Evaluating model...")
    scores = evaluator(model)
    logging.info(f"Final evaluation scores: {scores}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Sentence Transformer model with custom parameters.")
    parser.add_argument("--data_path", type=str, default="data/processed_dataset",
                        help="Path to the processed dataset (used if --data_paths is not provided).")
    parser.add_argument("--data_paths", type=str, nargs="+",
                        help="List of paths to processed datasets to combine by interleaving. If provided, overrides data_path.")
    parser.add_argument("--model_name", type=str, default="intfloat/multilingual-e5-large",
                        help="Name of the pre-trained model to load.")
    parser.add_argument("--output_dir", type=str, default="models_tmp/e5-large-tmp",
                        help="Directory to save model checkpoints.")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training and evaluation.")
    parser.add_argument("--learning_rate", type=float, default=2e-6,
                        help="Learning rate for training.")
    parser.add_argument("--e5_prefix", action="store_true", default=False,
                        help="Add prefixes to the dataset for E5 model training.")
    
    args = parser.parse_args()
    
    main(data_path=args.data_path,
         data_paths=args.data_paths,
         model_name=args.model_name,
         output_dir=args.output_dir,
         num_train_epochs=args.num_train_epochs,
         batch_size=args.batch_size,
         learning_rate=args.learning_rate)
