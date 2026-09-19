import json
import logging
import yaml
import os
import string
import re
from openai import OpenAI
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

def setup_logging():
    logger = logging.getLogger("evaluate_stability")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler("evaluate_stability.log", encoding="utf-8")
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

def normalize_answer(s):
    if not isinstance(s, str):
        return ""
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

def is_exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def main():
    try:
        config = load_config()
        cfg_exp = config.get("explanation", {})
        cfg_eval = config.get("evaluation", {})
        
        base_jsonl = cfg_exp.get("base_explanations_jsonl", "data/processed/base_explanations.jsonl")
        pert_jsonl = cfg_exp.get("perturbation_results_jsonl", "data/processed/perturbation_results.jsonl")
        output_json = cfg_eval.get("stability_results_json", "data/processed/stability_results.json")
        model_name = cfg_eval.get("judge_model", "openai/gpt-4o")
        
        client = get_openai_client()
        
        base_cases = {}
        with open(base_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                base_cases[data["id"]] = data
                
        perturbations = []
        with open(pert_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                perturbations.append(json.loads(line.strip()))
                
        logger.info(f"Loaded {len(base_cases)} base cases and {len(perturbations)} perturbed rules.")
        
        eval_prompt_template = """Compare these two explanations and determine if they use the same logical reasoning to arrive at the answer. Output exactly 'True' if the logic is fundamentally the same, or 'False' if the logic is different.
Explanation 1 (Base): {base_rule}
Explanation 2 (Perturbed): {perturbed_rule}
Output (True/False):"""
        
        results = []
        metrics = defaultdict(lambda: {
            "total": 0, 
            "rule_stable": 0, 
            "answer_stable": 0,
            "answer_stable_but_rule_changed": 0,
            "conditional_rule_stable": 0
        })
        
        for idx, pert in enumerate(perturbations):
            q_id = pert["id"]
            pert_type = pert["perturbation_type"]
            
            base_case = base_cases.get(q_id, {})
            base_rule = base_case.get("base_explanation", "")
            base_generated_answer = base_case.get("base_generated_answer", "")
            gold_answer = base_case.get("true_answer", "")
            
            pert_rule = pert.get("explanation", "")
            pert_answer = pert.get("generated_answer", "")
            
            answer_correct = is_exact_match(pert_answer, gold_answer)
            answer_same_as_baseline = is_exact_match(pert_answer, base_generated_answer)
            
            if base_rule.strip() == pert_rule.strip():
                is_rule_stable = True
            else:
                prompt = eval_prompt_template.format(
                    base_rule=base_rule,
                    perturbed_rule=pert_rule
                )
                
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                
                output = resp.choices[0].message.content.strip()
                is_rule_stable = output.lower() == "true" or "true" in output.lower()
                
            pert["answer_correct"] = answer_correct
            pert["answer_same_as_baseline"] = answer_same_as_baseline
            pert["is_stable"] = is_rule_stable
            results.append(pert)
            
            for category in ["overall", pert_type]:
                metrics[category]["total"] += 1
                if is_rule_stable:
                    metrics[category]["rule_stable"] += 1
                if answer_same_as_baseline:
                    metrics[category]["answer_stable"] += 1
                    if not is_rule_stable:
                        metrics[category]["answer_stable_but_rule_changed"] += 1
                    else:
                        metrics[category]["conditional_rule_stable"] += 1
                
            if (idx + 1) % 50 == 0:
                logger.info(f"Evaluated {idx + 1}/{len(perturbations)} rules...")
                
        logger.info("Evaluation Complete!")
        
        summary = {}
        for category, stats in metrics.items():
            total = stats["total"]
            r_stable = stats["rule_stable"]
            a_stable = stats["answer_stable"]
            cond_r_stable = stats["conditional_rule_stable"]
            a_stable_r_changed = stats["answer_stable_but_rule_changed"]
            
            summary[category] = {
                "total": total,
                "rule_stable_count": r_stable,
                "rule_stability_percentage": round((r_stable / total) * 100, 2) if total > 0 else 0,
                "answer_stable_count": a_stable,
                "answer_stability_percentage": round((a_stable / total) * 100, 2) if total > 0 else 0,
                "same_answer_changed_rule_count": a_stable_r_changed,
                "same_answer_changed_rule_percentage": round((a_stable_r_changed / total) * 100, 2) if total > 0 else 0,
                "conditional_rule_stability_given_answer_stable": round((cond_r_stable / a_stable) * 100, 2) if a_stable > 0 else 0
            }
            logger.info(f"[{category.upper()}] Rule Stability: {r_stable}/{total} ({summary[category]['rule_stability_percentage']}%)")
            
        final_output = {
            "summary_metrics": summary,
            "evaluated_perturbations": results
        }
        
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
            
        logger.info(f"Saved evaluation results to {output_json}")
        
    except Exception as e:
        logger.exception("An error occurred during stability evaluation.")

if __name__ == "__main__":
    main()
