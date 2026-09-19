import json
import logging
import yaml
import os
import re
import string
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def setup_logging():
    logger = logging.getLogger("generate_answer")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler("generate_answer.log")
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

def get_openai_client():
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    if openrouter_key and openrouter_key != "your_openrouter_api_key_here":
        logger.info("Using OpenRouter API key.")
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
        )
    elif openai_key and openai_key != "your_openai_api_key_here":
        logger.info("Using OpenAI API key.")
        return OpenAI(api_key=openai_key)
    else:
        # Fallback check for api_key.txt for backward compatibility
        if os.path.exists("api_key.txt"):
            with open("api_key.txt", "r") as f:
                key = f.read().strip()
                if key and not key.startswith("PASTE"):
                    logger.info("Using API key from api_key.txt")
                    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
                    
        raise ValueError("No valid OPENAI_API_KEY or OPENROUTER_API_KEY found in environment or .env file.")

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = set(prediction_tokens) & set(ground_truth_tokens)
    num_same = len(common)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def main():
    try:
        config = load_config()
        cfg_gen = config.get("generator", {})
        cfg_data = config.get("data", {})
        
        results_jsonl = cfg_data.get("retrieval_results", "data/processed/retrieval_results.jsonl")
        baseline_out = cfg_gen.get("baseline_answers_jsonl", "data/processed/baseline_answers.jsonl")
        incorrect_out = cfg_gen.get("incorrect_baselines_jsonl", "data/processed/incorrect_baselines.jsonl")
        
        model_name = cfg_gen.get("model_name", "openai/gpt-4o-mini")
        temperature = cfg_gen.get("temperature", 0.0)
        
        input_json = cfg_data.get("output_json", "data/processed/selected_hotpotqa.json")
        with open(input_json, "r", encoding="utf-8") as f:
            selected_data = json.load(f)
            
        ground_truths = {item["id"]: item["answer"] for item in selected_data}
        client = get_openai_client()
        
        logger.info(f"Loading retrieval results from {results_jsonl}...")
        valid_retrievals = []
        with open(results_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                if item.get("all_supporting_found", False):
                    valid_retrievals.append(item)
                    
        logger.info(f"Found {len(valid_retrievals)} questions where all supporting facts were retrieved.")
        
        baseline_cases = []
        incorrect_cases = []
        
        prompt_template = """Answer the question using only the supplied documents.
Give a short factual answer. Do not write full sentences, return only the exact entity or phrase.

Question:
{question}

Documents:
{documents}"""
        
        for idx, item in enumerate(valid_retrievals):
            q_id = item["id"]
            question = item["question"]
            retrieved_docs = item["retrieved_documents"]
            
            docs_text = ""
            for doc in retrieved_docs:
                docs_text += f"Document: {doc['title']}\n{doc['text']}\n\n"
                
            prompt = prompt_template.format(question=question, documents=docs_text.strip())
            logger.info(f"Processing question {idx+1}/{len(valid_retrievals)}: {question}")
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            
            predicted_answer = response.choices[0].message.content.strip()
            true_answer = ground_truths[q_id]
            
            em = exact_match_score(predicted_answer, true_answer)
            f1 = f1_score(predicted_answer, true_answer)
            substring_match = normalize_answer(true_answer) in normalize_answer(predicted_answer)
            
            result_item = {
                "id": q_id,
                "question": question,
                "true_answer": true_answer,
                "predicted_answer": predicted_answer,
                "exact_match": em,
                "f1_score": f1,
                "retrieved_documents": retrieved_docs
            }
            
            if em or f1 > 0.5 or substring_match:
                logger.info(f"Correct! True: '{true_answer}' | Pred: '{predicted_answer}' (F1: {f1:.2f})")
                baseline_cases.append(result_item)
            else:
                logger.warning(f"Incorrect. True: '{true_answer}' | Pred: '{predicted_answer}' (F1: {f1:.2f})")
                incorrect_cases.append(result_item)
                
        logger.info(f"Finished generation. Valid baselines: {len(baseline_cases)}, Incorrect: {len(incorrect_cases)}")
        
        with open(baseline_out, "w", encoding="utf-8") as f:
            for case in baseline_cases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
                
        with open(incorrect_out, "w", encoding="utf-8") as f:
            for case in incorrect_cases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
                
        logger.info(f"Saved {len(baseline_cases)} valid baselines to {baseline_out}")
        
    except Exception as e:
        logger.exception("An error occurred during generation.")

if __name__ == "__main__":
    main()
