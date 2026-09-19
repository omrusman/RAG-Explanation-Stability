import os
import json
import logging
import pandas as pd
import yaml
import random
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

def setup_logging():
    logger = logging.getLogger("prepare_data")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler("prepare_data.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

logger = setup_logging()

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

def create_directories():
    dirs = [
        "data/raw",
        "data/processed",
        "results/answers",
        "results/rules",
        "results/metrics",
        "results/figures",
        "reports/figures",
        "reports/diagrams",
        "src"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info("Ensured necessary project directories exist.")

def select_candidates(dataset, num_candidates=50, seed=42):
    random.seed(seed)
    candidates = []
    
    for item in dataset:
        answer = item.get("answer", "").strip().lower()
        if answer in ["yes", "no"]:
            continue
            
        if len(answer.split()) > 5:
            continue
            
        supp_facts = item.get("supporting_facts", {})
        titles = supp_facts.get("title", [])
        unique_titles = list(set(titles))
        
        if len(unique_titles) != 2:
            continue
            
        candidates.append(item)
        
    logger.info(f"Found {len(candidates)} total candidates matching criteria.")
    
    if len(candidates) < num_candidates:
        logger.warning(f"Only found {len(candidates)} candidates, which is less than requested {num_candidates}.")
        return candidates
        
    selected = random.sample(candidates, num_candidates)
    return selected

def main():
    try:
        create_directories()
        config = load_config()
        cfg_data = config.get("data", {})
        
        dataset_name = cfg_data.get("dataset_name", "hotpot_qa")
        dataset_config = cfg_data.get("dataset_config", "distractor")
        split = cfg_data.get("split", "validation")
        output_json = cfg_data.get("output_json", "data/processed/selected_hotpotqa.json")
        output_csv = cfg_data.get("output_csv", "data/processed/candidates_review.csv")
        num_candidates = cfg_data.get("num_candidates", 50)
        seed = config.get("project", {}).get("seed", 42)
        
        logger.info(f"Loading dataset {dataset_name} ({dataset_config}), split: {split}")
        try:
            dataset = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=True)
        except TypeError:
            dataset = load_dataset(dataset_name, dataset_config, split=split)
            
        logger.info(f"Loaded {len(dataset)} examples from {split} split.")
        logger.info("Selecting candidate questions...")
        selected = select_candidates(dataset, num_candidates, seed)
        
        logger.info(f"Saving {len(selected)} candidates to {output_json}")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(selected, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Creating review CSV at {output_csv}")
        review_data = []
        for item in selected:
            supp_facts = item.get("supporting_facts", {})
            titles = list(set(supp_facts.get("title", [])))
            review_data.append({
                "id": item.get("id"),
                "question": item.get("question"),
                "answer": item.get("answer"),
                "supporting_titles": ", ".join(titles)
            })
            
        df = pd.DataFrame(review_data)
        df.to_csv(output_csv, index=False)
        logger.info("Data preparation completed successfully.")
        
    except Exception as e:
        logger.exception("An error occurred during data preparation.")

if __name__ == "__main__":
    main()
