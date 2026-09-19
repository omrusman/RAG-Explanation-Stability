import streamlit as st
import json
import pandas as pd
import os

st.set_page_config(page_title="RAG Explanation Stability Dashboard", layout="wide", page_icon="🧩")

def load_data():
    base_cases = []
    base_file = "data/processed/base_explanations.jsonl"
    if os.path.exists(base_file):
        with open(base_file, "r", encoding="utf-8") as f:
            for line in f:
                base_cases.append(json.loads(line.strip()))
                
    perturbations = []
    metrics = None
    eval_file = "data/processed/stability_results.json"
    pert_file = "data/processed/perturbation_results.jsonl"
    
    if os.path.exists(eval_file):
        with open(eval_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            perturbations = data.get("evaluated_perturbations", [])
            metrics = data.get("summary_metrics", {})
    elif os.path.exists(pert_file):
        with open(pert_file, "r", encoding="utf-8") as f:
            for line in f:
                perturbations.append(json.loads(line.strip()))
                
    return base_cases, perturbations, metrics

st.title("🧩 RAG Explanation Stability Dashboard")
st.markdown("Analyze how the LLM's explanation rules change when the context or question is perturbed.")

base_cases, perturbations, metrics = load_data()

if metrics:
    st.divider()
    st.subheader("📊 Final Stability Metrics")
    
    cats = ["overall", "deletion", "reordering", "paraphrase"]
    
    st.markdown("**Rule Stability (Does the explanation logic hold?)**")
    r_cols = st.columns(4)
    for i, cat in enumerate(cats):
        if cat in metrics:
            r_cols[i].metric(
                f"{cat.capitalize()} Rule Stability", 
                f"{metrics[cat]['rule_stability_percentage']}%", 
                f"{metrics[cat]['rule_stable_count']} / {metrics[cat]['total']}",
                delta_color="off"
            )
            
    st.markdown("**Answer Stability (Does the generated answer hold?)**")
    a_cols = st.columns(4)
    for i, cat in enumerate(cats):
        if cat in metrics:
            a_cols[i].metric(
                f"{cat.capitalize()} Answer Stability", 
                f"{metrics[cat]['answer_stability_percentage']}%", 
                f"{metrics[cat]['answer_stable_count']} / {metrics[cat]['total']}",
                delta_color="off"
            )
            
    st.markdown("**Same Answer + Changed Rule (Hidden Instability)**")
    c_cols = st.columns(4)
    for i, cat in enumerate(cats):
        if cat in metrics:
            c_cols[i].metric(
                f"{cat.capitalize()} Hidden Instability", 
                f"{metrics[cat]['same_answer_changed_rule_percentage']}%", 
                f"{metrics[cat]['same_answer_changed_rule_count']} / {metrics[cat]['total']}",
                delta_color="inverse"
            )

if not base_cases:
    st.warning("No base explanations found. Run the pipeline first or ensure data/processed/base_explanations.jsonl is present.")
    st.stop()

st.divider()
st.subheader("🔍 Explore Question Stability & Perturbations")

question_options = {case["id"]: case["question"] for case in base_cases}
selected_id = st.selectbox("Select a Question to Inspect:", options=list(question_options.keys()), format_func=lambda x: question_options[x])

selected_base = next((c for c in base_cases if c["id"] == selected_id), None)
selected_perts = [p for p in perturbations if p["id"] == selected_id]

if selected_base:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Base Context")
        st.markdown(f"**Original Question:** {selected_base['question']}")
        st.markdown(f"**True Answer (Gold):** `{selected_base['true_answer']}`")
        
        base_ans = selected_base.get('base_generated_answer', selected_base.get('predicted_answer', ''))
        st.markdown(f"**Base Generated Answer:** `{base_ans}`")
        
        with st.expander("View Retrieved Documents (Base Context)"):
            for i, doc in enumerate(selected_base['retrieved_documents']):
                st.markdown(f"**[{i}] {doc['title']}**")
                st.caption(doc['text'])
    
    with col2:
        st.markdown("### Base Explanation Rule")
        st.info(selected_base.get('base_explanation', 'No base explanation found.'))

st.divider()
st.subheader("Perturbation Results & Comparison")

if not selected_perts:
    st.info("No perturbations generated for this question yet.")
else:
    pert_types = list(set([p['perturbation_type'] for p in selected_perts]))
    selected_type = st.radio("Select Perturbation Type to Compare:", pert_types, horizontal=True)
    
    filtered_perts = [p for p in selected_perts if p['perturbation_type'] == selected_type]
    
    st.markdown("---")
    
    for pert in filtered_perts:
        st.markdown("#### Perturbation Trial Details")
        
        if selected_type == "deletion":
            st.markdown(f"**Documents Included:** {pert.get('doc_combination', [])}")
        elif selected_type == "reordering":
            st.markdown(f"**Document Sequence Order:** {pert.get('doc_combination', [])}")
        elif selected_type == "paraphrase":
            st.markdown(f"**Paraphrased Question:** {pert.get('question_used', '')}")
            
        col_base, col_pert = st.columns(2)
        with col_base:
            st.success("**Base Answer:** " + str(selected_base.get('base_generated_answer', '')))
            st.success("**Base Rule:**\n\n" + str(selected_base.get('base_explanation', '')))
            
        with col_pert:
            pert_ans = pert.get('generated_answer', '')
            pert_rule = pert.get('explanation', '').strip()
            
            ans_stable = pert.get("answer_same_as_baseline")
            rule_stable = pert.get("is_stable")
            
            if ans_stable is True:
                st.info(f"**Perturbed Answer (Same as Base ✅):** {pert_ans}")
            else:
                st.error(f"**Perturbed Answer (Changed ❌):** {pert_ans}")
                
            if rule_stable is True:
                st.info(f"**Perturbed Rule (Stable ✅):**\n\n{pert_rule}")
            elif rule_stable is False:
                st.error(f"**Perturbed Rule (Changed ❌):**\n\n{pert_rule}")
            else:
                st.warning(f"**Perturbed Rule (Not Evaluated):**\n\n{pert_rule}")
        
        st.markdown("---")
