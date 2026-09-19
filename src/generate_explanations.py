import json
import logging
import yaml
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def setup_logging():
    logger = logging.getLogger("generate_explanations")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler("generate_explanations.log", encoding="utf-8")
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

def main():
    try:
        config = load_config()
        cfg_gen = config.get("generator", {})
        cfg_exp = config.get("explanation", {})
        
        input_jsonl = cfg_gen.get("baseline_answers_jsonl", "data/processed/baseline_answers.jsonl")
        output_jsonl = cfg_exp.get("base_explanations_jsonl", "data/processed/base_explanations.jsonl")
        
        model_name = cfg_gen.get("model_name", "openai/gpt-4o-mini")
        temperature = cfg_gen.get("temperature", 0.0)
        
        client = get_openai_client()
        
        logger.info(f"Loading baseline cases from {input_jsonl}...")
        baselines = []
        with open(input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                baselines.append(json.loads(line.strip()))
                
        logger.info(f"Loaded {len(baselines)} baseline cases.")
        
        explanation_prompt_template = """Based on the documents, answer the question and explain why the answer is correct with a short, logical rule or reason.
Output your response as a valid JSON object with exactly two keys: "answer" and "explanation".
Question: {question}
Documents:
{documents}"""
        
        results = []
        for idx, item in enumerate(baselines):
            q_id = item["id"]
            question = item["question"]
            retrieved_docs = item["retrieved_documents"]
            
            docs_text = ""
            for doc in retrieved_docs:
                docs_text += f"Document: {doc['title']}\n{doc['text']}\n\n"
                
            prompt = explanation_prompt_template.format(
                question=question, 
                documents=docs_text.strip()
            )
            
            logger.info(f"Processing question {idx+1}/{len(baselines)}: {question}")
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            
            response_text = response.choices[0].message.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:-3].strip()
                
            try:
                parsed_json = json.loads(response_text)
                base_answer = str(parsed_json.get("answer", ""))
                base_explanation = str(parsed_json.get("explanation", ""))
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON for ID {q_id}. Raw text: {response_text}")
                base_answer = "Error parsing answer"
                base_explanation = "Error parsing explanation"
            
            result_item = item.copy()
            result_item["base_generated_answer"] = base_answer
            result_item["base_explanation"] = base_explanation
            results.append(result_item)
            
            logger.info(f"Answer snippet: {base_answer[:50]} | Rule snippet: {base_explanation[:50]}...")
                
        logger.info(f"Finished generating explanations for {len(results)} cases.")
        
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
        logger.info(f"Saved explanations to {output_jsonl}")
        
    except Exception as e:
        logger.exception("An error occurred during explanation generation.")

if __name__ == "__main__":
    main()
