import json
import sys

# Ensure UTF-8 output encoding for console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def load_data():
    with open("data/processed/stability_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        metrics = data.get("summary_metrics", {})
    return metrics

def main():
    metrics = load_data()
    print("\n==========================================")
    print("RAG Explanation Stability Metrics Summary")
    print("==========================================")
    for cat, values in metrics.items():
        print(f"\n--- {cat.upper()} ---")
        print(f"  Total Trials: {values['total']}")
        print(f"  Rule Stability: {values['rule_stability_percentage']}% ({values['rule_stable_count']}/{values['total']})")
        print(f"  Answer Stability: {values['answer_stability_percentage']}% ({values['answer_stable_count']}/{values['total']})")
        print(f"  Hidden Instability (Same Answer, Changed Rule): {values['same_answer_changed_rule_percentage']}% ({values['same_answer_changed_rule_count']}/{values['total']})")
        print(f"  Conditional Rule Stability (Given Answer Stable): {values['conditional_rule_stability_given_answer_stable']}%")

if __name__ == "__main__":
    main()
