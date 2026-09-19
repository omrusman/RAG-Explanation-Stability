import json
import logging
import yaml
import os
import itertools
import random
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def setup_logging():
    logger = logging.getLogger("run_perturbations")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler("run_perturbations.log", encoding="utf-8")
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
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)
    elif openai_key and openai_key != "your_openai_api_key_here":
        logger.info("Using OpenAI API key.")
        return OpenAI(api_key=openai_key)
    else:
        if os.path.exists("api_key.txt"):
            with open("api_key.txt", "r") as f:
                key = f.read().strip()
                if key and not key.startswith("PASTE"):
                    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        raise ValueError("No valid OPENAI_API_KEY or OPENROUTER_API_KEY found in environment or .env file.")

def format_docs(docs):
    docs_text = ""
    for doc in docs:
        docs_text += f"Document: {doc['title']}\n{doc['text']}\n\n"
    return docs_text.strip()

def parse_json_response(response_text, q_id):
    if response_text.startswith("```json"):
        response_text = response_text[7:-3].strip()
    elif response_text.startswith("```"):
        response_text = response_text[3:-3].strip()
        
    try:
        parsed_json = json.loads(response_text)
        return str(parsed_json.get("answer", "")), str(parsed_json.get("explanation", ""))
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON for ID {q_id}. Raw text: {response_text}")
        return "Error parsing answer", "Error parsing explanation"

def main():
    try:
        config = load_config()
        cfg_gen = config.get("generator", {})
        cfg_exp = config.get("explanation", {})
        
        input_jsonl = cfg_exp.get("base_explanations_jsonl", "data/processed/base_explanations.jsonl")
        output_jsonl = cfg_exp.get("perturbation_results_jsonl", "data/processed/perturbation_results.jsonl")
        
        model_name = cfg_gen.get("model_name", "openai/gpt-4o-mini")
        temperature = cfg_gen.get("temperature", 0.0)
        
        client = get_openai_client()
        
        logger.info(f"Loading base explanations from {input_jsonl}...")
        base_cases = []
        with open(input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                base_cases.append(json.loads(line.strip()))
                
        logger.info(f"Loaded {len(base_cases)} cases.")
        
        explanation_prompt_template = """Based on the documents, answer the question and explain why the answer is correct with a short, logical rule or reason.
Output your response as a valid JSON object with exactly two keys: "answer" and "explanation".
Question: {question}
Documents:
{documents}"""

        paraphrase_prompt_template = """Paraphrase the following question. Keep the exact same meaning, but change the wording. Return ONLY the paraphrased question.
Original Question: {question}
Paraphrased Question {n}:"""
        
        results = []
        random.seed(42)
        
        for idx, item in enumerate(base_cases):
            q_id = item["id"]
            question = item["question"]
            retrieved_docs = item["retrieved_documents"]
            
            logger.info(f"--- Processing question {idx+1}/{len(base_cases)}: {question} ---")
            
            # Deletion (14 combinations)
            logger.info("Running Document Deletion (14 combinations)...")
            doc_indices = [0, 1, 2, 3]
            combinations = []
            for r in range(1, 4):
                combinations.extend(list(itertools.combinations(doc_indices, r)))
                
            for combo in combinations:
                combo_docs = [retrieved_docs[i] for i in combo]
                prompt = explanation_prompt_template.format(
                    question=question, 
                    documents=format_docs(combo_docs)
                )
                
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                generated_ans, generated_rule = parse_json_response(resp.choices[0].message.content.strip(), q_id)
                
                results.append({
                    "id": q_id,
                    "perturbation_type": "deletion",
                    "doc_combination": combo,
                    "question_used": question,
                    "generated_answer": generated_ans,
                    "explanation": generated_rule
                })
                
            # Context Reordering (3 permutations)
            logger.info("Running Context Reordering (3 permutations)...")
            orderings = [
                [3, 2, 1, 0],
                random.sample([0, 1, 2, 3], 4),
                random.sample([0, 1, 2, 3], 4)
            ]
            for i in range(1, 3):
                while orderings[i] == [0, 1, 2, 3] or orderings[i] == [3, 2, 1, 0] or (i == 2 and orderings[1] == orderings[2]):
                    orderings[i] = random.sample([0, 1, 2, 3], 4)

            for order in orderings:
                ordered_docs = [retrieved_docs[i] for i in order]
                prompt = explanation_prompt_template.format(
                    question=question, 
                    documents=format_docs(ordered_docs)
                )
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                generated_ans, generated_rule = parse_json_response(resp.choices[0].message.content.strip(), q_id)
                
                results.append({
                    "id": q_id,
                    "perturbation_type": "reordering",
                    "doc_combination": order, 
                    "question_used": question,
                    "generated_answer": generated_ans,
                    "explanation": generated_rule
                })
            
            # Paraphrasing (2 paraphrases)
            logger.info("Running Question Paraphrasing (2 variations)...")
            for para_n in range(1, 3):
                para_prompt = paraphrase_prompt_template.format(question=question, n=para_n)
                para_resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": para_prompt}],
                    temperature=0.7 if para_n == 2 else temperature,
                )
                paraphrased_q = para_resp.choices[0].message.content.strip()
                
                prompt = explanation_prompt_template.format(
                    question=paraphrased_q, 
                    documents=format_docs(retrieved_docs)
                )
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                generated_ans, generated_rule = parse_json_response(resp.choices[0].message.content.strip(), q_id)
                
                results.append({
                    "id": q_id,
                    "perturbation_type": "paraphrase",
                    "doc_combination": [0, 1, 2, 3], 
                    "question_used": paraphrased_q,
                    "original_question": question,
                    "generated_answer": generated_ans,
                    "explanation": generated_rule
                })
            
            logger.info(f"Finished processing ID: {q_id}. Total results generated so far: {len(results)}")
            
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
        logger.info(f"Finished ALL perturbations. Saved {len(results)} rules to {output_jsonl}")
        
    except Exception as e:
        logger.exception("An error occurred during perturbation generation.")

if __name__ == "__main__":
    main()
