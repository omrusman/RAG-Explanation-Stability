import json
import logging
import yaml
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

def setup_logging():
    logger = logging.getLogger("build_index")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler("build_index.log")
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
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def extract_documents(selected_data):
    unique_docs = {}
    for item in selected_data:
        context_dict = item.get("context", {})
        titles = context_dict.get("title", [])
        sentences_list = context_dict.get("sentences", [])
        
        for title, sentences in zip(titles, sentences_list):
            if title not in unique_docs:
                unique_docs[title] = {
                    "id": title,
                    "title": title,
                    "text": " ".join(sentences)
                }
    
    docs_list = list(unique_docs.values())
    logger.info(f"Extracted {len(docs_list)} unique documents from {len(selected_data)} questions.")
    return docs_list

def main():
    try:
        config = load_config()
        cfg_data = config.get("data", {})
        cfg_ret = config.get("retriever", {})
        
        input_json = cfg_data.get("output_json", "data/processed/selected_hotpotqa.json")
        docs_jsonl = cfg_data.get("documents_jsonl", "data/processed/documents.jsonl")
        model_name = cfg_ret.get("model_name", "all-MiniLM-L6-v2")
        index_path = cfg_ret.get("index_path", "data/processed/faiss_index.bin")
        
        logger.info(f"Loading selected questions from {input_json}...")
        with open(input_json, "r", encoding="utf-8") as f:
            selected_data = json.load(f)
            
        logger.info("Extracting unique documents from context...")
        docs = extract_documents(selected_data)
        
        logger.info(f"Saving documents to {docs_jsonl}...")
        with open(docs_jsonl, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                
        logger.info(f"Loading embedding model: {model_name}...")
        model = SentenceTransformer(model_name)
        
        texts_to_embed = [f"{d['title']}. {d['text']}" for d in docs]
        logger.info(f"Encoding {len(texts_to_embed)} documents...")
        embeddings = model.encode(texts_to_embed, show_progress_bar=True, convert_to_numpy=True)
        
        logger.info("Building FAISS index...")
        dimension = embeddings.shape[1]
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        logger.info(f"Index built with {index.ntotal} vectors of dimension {dimension}.")
        logger.info(f"Saving FAISS index to {index_path}...")
        faiss.write_index(index, index_path)
        logger.info("Build index completed successfully.")
        
    except Exception as e:
        logger.exception("An error occurred during index building.")

if __name__ == "__main__":
    main()
