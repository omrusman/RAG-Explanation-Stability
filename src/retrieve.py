import json
import logging
import yaml
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

def setup_logging():
    logger = logging.getLogger("retrieve")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler("retrieve.log")
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

def load_documents(docs_jsonl):
    docs = []
    with open(docs_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line.strip()))
    return docs

def main():
    try:
        config = load_config()
        cfg_data = config.get("data", {})
        cfg_ret = config.get("retriever", {})
        
        input_json = cfg_data.get("output_json", "data/processed/selected_hotpotqa.json")
        docs_jsonl = cfg_data.get("documents_jsonl", "data/processed/documents.jsonl")
        results_jsonl = cfg_data.get("retrieval_results", "data/processed/retrieval_results.jsonl")
        
        model_name = cfg_ret.get("model_name", "all-MiniLM-L6-v2")
        index_path = cfg_ret.get("index_path", "data/processed/faiss_index.bin")
        top_k = cfg_ret.get("top_k", 4)
        
        logger.info(f"Loading selected questions from {input_json}...")
        with open(input_json, "r", encoding="utf-8") as f:
            selected_data = json.load(f)
            
        logger.info(f"Loading documents from {docs_jsonl}...")
        docs = load_documents(docs_jsonl)
        idx_to_doc = {i: d for i, d in enumerate(docs)}
        
        logger.info(f"Loading FAISS index from {index_path}...")
        index = faiss.read_index(index_path)
        
        logger.info(f"Loading embedding model: {model_name}...")
        model = SentenceTransformer(model_name)
        
        queries = [item["question"] for item in selected_data]
        query_embeddings = model.encode(queries, show_progress_bar=True, convert_to_numpy=True)
        faiss.normalize_L2(query_embeddings)
        
        scores, indices = index.search(query_embeddings, top_k)
        
        results = []
        correct_retrieval_count = 0
        
        for i, item in enumerate(selected_data):
            q_scores = scores[i]
            q_indices = indices[i]
            
            retrieved_docs = []
            retrieved_titles = []
            
            for j in range(top_k):
                doc_idx = q_indices[j]
                doc = idx_to_doc[doc_idx]
                retrieved_titles.append(doc["title"])
                retrieved_docs.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "text": doc["text"],
                    "score": float(q_scores[j])
                })
            
            supp_facts = item.get("supporting_facts", {})
            true_titles = set(supp_facts.get("title", []))
            retrieved_true_titles = true_titles.intersection(set(retrieved_titles))
            all_found = len(retrieved_true_titles) == len(true_titles)
            
            if all_found:
                correct_retrieval_count += 1
                
            results.append({
                "id": item["id"],
                "question": item["question"],
                "retrieved_documents": retrieved_docs,
                "all_supporting_found": all_found
            })
            
        accuracy = (correct_retrieval_count / len(selected_data)) * 100
        logger.info(f"Retrieval Accuracy (all supporting facts found): {accuracy:.2f}% ({correct_retrieval_count}/{len(selected_data)})")
        
        with open(results_jsonl, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
        logger.info("Retrieval completed successfully.")
        
    except Exception as e:
        logger.exception("An error occurred during retrieval.")

if __name__ == "__main__":
    main()
