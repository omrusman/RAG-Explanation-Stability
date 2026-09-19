import json
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("reports/figures", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

def load_data():
    base_cases = {}
    base_path = "data/processed/base_explanations.jsonl"
    if os.path.exists(base_path):
        with open(base_path, "r", encoding="utf-8") as f:
            for line in f:
                case = json.loads(line.strip())
                base_cases[case["id"]] = case

    eval_path = "data/processed/stability_results.json"
    perturbations = []
    metrics = {}
    if os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            perturbations = data.get("evaluated_perturbations", [])
            metrics = data.get("summary_metrics", {})
        
    return base_cases, perturbations, metrics

def create_excel_report(base_cases, perturbations, metrics):
    print("Generating Excel Report...")
    rows = []
    for pert in perturbations:
        base = base_cases.get(pert["id"], {})
        row = {
            "Question ID": pert.get("id"),
            "Perturbation Type": pert.get("perturbation_type", "").capitalize(),
            "Original Question": base.get("question"),
            "True Answer (Gold)": base.get("true_answer"),
            "Base Predicted Answer": base.get("base_generated_answer", base.get("predicted_answer")),
            "Base Explanation": base.get("base_explanation"),
            "Configuration Details": str(pert.get("doc_combination", pert.get("question_used", ""))),
            "Perturbed Predicted Answer": pert.get("generated_answer"),
            "Perturbed Explanation": pert.get("explanation"),
            "Answer Stable (Exact Match)": pert.get("answer_same_as_baseline"),
            "Rule Stable (LLM Judged)": pert.get("is_stable"),
            "Hidden Instability (Same Answer, Changed Rule)": (pert.get("answer_same_as_baseline") is True) and (pert.get("is_stable") is False)
        }
        rows.append(row)
    df_raw = pd.DataFrame(rows)
    
    summary_rows = []
    cats = ["overall", "deletion", "reordering", "paraphrase"]
    labels = ["Overall", "Deletion", "Reordering", "Paraphrasing"]
    for i, cat in enumerate(cats):
        if cat in metrics:
            summary_rows.append({
                "Category": labels[i],
                "Total Trials": metrics[cat]["total"],
                "Rule Stability (%)": metrics[cat]["rule_stability_percentage"],
                "Answer Stability (%)": metrics[cat]["answer_stability_percentage"],
                "Hidden Instability (%)": metrics[cat]["same_answer_changed_rule_percentage"]
            })
    df_summary = pd.DataFrame(summary_rows)

    excel_path = "reports/experiment_data_updated.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name="Summary Metrics", index=False)
        df_raw.to_excel(writer, sheet_name="Raw Data", index=False)
        
    print(f"Excel report saved to {excel_path}")

def generate_graphs(metrics):
    if not metrics:
        print("No metrics available for graph generation.")
        return
    print("Generating Graphs...")
    sns.set_theme(style="whitegrid")
    
    cats = ["overall", "deletion", "reordering", "paraphrase"]
    labels = ["Overall", "Deletion", "Reordering", "Paraphrasing"]
    
    rule_scores = [metrics.get(cat, {}).get("rule_stability_percentage", 0) for cat in cats]
    ans_scores = [metrics.get(cat, {}).get("answer_stability_percentage", 0) for cat in cats]
    hidden_scores = [metrics.get(cat, {}).get("same_answer_changed_rule_percentage", 0) for cat in cats]
    
    x = range(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar([p - width/2 for p in x], ans_scores, width, label='Answer Stability %', color='#3498db')
    rects2 = ax.bar([p + width/2 for p in x], rule_scores, width, label='Rule Stability %', color='#2ecc71')
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Answer Stability vs. Rule Stability by Perturbation Type')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 105)
    
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom')
                        
    fig.tight_layout()
    plt.savefig("reports/figures/stability_comparison.png", dpi=300)
    plt.savefig("results/figures/stability_comparison.png", dpi=300)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    rects = ax.bar(labels, hidden_scores, color='#e74c3c', width=0.5)
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Hidden Instability (Same Answer, Changed Rule)')
    ax.set_ylim(0, max(hidden_scores) + 5 if len(hidden_scores) > 0 and max(hidden_scores) > 0 else 10)
    
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom')
                    
    fig.tight_layout()
    plt.savefig("reports/figures/hidden_instability.png", dpi=300)
    plt.savefig("results/figures/hidden_instability.png", dpi=300)
    plt.close()
    print("Graphs generated successfully.")

def generate_html_report(metrics):
    if not metrics:
        return
    print("Generating HTML Report...")
    table_html = """
    <table>
        <tr>
            <th>Category</th>
            <th>Total Trials</th>
            <th>Rule Stability</th>
            <th>Answer Stability</th>
            <th>Hidden Instability</th>
        </tr>
    """
    cats = ["overall", "deletion", "reordering", "paraphrase"]
    labels = ["Overall", "Deletion", "Reordering", "Paraphrasing"]
    for i, cat in enumerate(cats):
        if cat in metrics:
            table_html += f"""
        <tr>
            <td><strong>{labels[i]}</strong></td>
            <td>{metrics[cat]['total']}</td>
            <td>{metrics[cat]['rule_stability_percentage']}%</td>
            <td>{metrics[cat]['answer_stability_percentage']}%</td>
            <td>{metrics[cat]['same_answer_changed_rule_percentage']}%</td>
        </tr>
            """
    table_html += "</table>"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Explanation Stability - Final Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; max-width: 900px; margin: 40px auto; color: #333; padding: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .header {{ text-align: center; border-bottom: 2px solid #eee; padding-bottom: 20px; margin-bottom: 40px; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-top: 20px; margin-bottom: 40px; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 40px; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f8f9fa; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>RAG Explanation Stability</h1>
            <p>Research Report & Findings</p>
        </div>

        <h2>1. Introduction & Core Findings</h2>
        <p>This report documents the final experimental findings on whether rule-based explanations in Retrieval-Augmented Generation (RAG) systems are as stable as the predictions they explain.</p>
        
        <h2>2. Quantitative Results</h2>
        {table_html}
        
        <center><img src="figures/stability_comparison.png" alt="Stability Comparison"></center>
        <center><img src="figures/hidden_instability.png" alt="Hidden Instability"></center>
    </body>
    </html>
    """
    
    with open("reports/final_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("HTML Report generated successfully at reports/final_report.html")

def main():
    base_cases, perturbations, metrics = load_data()
    create_excel_report(base_cases, perturbations, metrics)
    generate_graphs(metrics)
    generate_html_report(metrics)

if __name__ == "__main__":
    main()
