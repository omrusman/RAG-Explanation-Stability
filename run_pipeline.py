#!/usr/bin/env python3
"""
RAG Explanation Stability - Master Pipeline Runner
Run all pipeline steps sequentially or execute individual pipeline stages.

Usage:
    python run_pipeline.py --step all
    python run_pipeline.py --step prepare
    python run_pipeline.py --step index
    python run_pipeline.py --step retrieve
    python run_pipeline.py --step answer
    python run_pipeline.py --step explain
    python run_pipeline.py --step perturb
    python run_pipeline.py --step evaluate
    python run_pipeline.py --step report
"""

import argparse
import sys
import subprocess
import os
from dotenv import load_dotenv

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

STEPS = [
    ("prepare", "src/prepare_data.py", "Stage 1: Preparing HotpotQA Dataset"),
    ("index", "src/build_index.py", "Stage 2: Building FAISS Vector Index"),
    ("retrieve", "src/retrieve.py", "Stage 3: Running Dense Retrieval"),
    ("answer", "src/generate_answer.py", "Stage 4: Generating Baseline Answers"),
    ("explain", "src/generate_explanations.py", "Stage 5: Generating Base Rule Explanations"),
    ("perturb", "src/run_perturbations.py", "Stage 6: Executing Perturbation Experiments"),
    ("evaluate", "src/evaluate_stability.py", "Stage 7: Evaluating Rule & Answer Stability"),
    ("report", "src/export_results.py", "Stage 8: Generating Reports and Visualizations")
]

def run_script(script_path, description):
    print(f"\n==================================================")
    print(f"Running {description}...")
    print(f"==================================================")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"Error: {script_path} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"Finished {description} successfully.")

def main():
    parser = argparse.ArgumentParser(description="RAG Explanation Stability Master Pipeline Runner")
    parser.add_argument(
        "--step",
        choices=["all"] + [s[0] for s in STEPS],
        default="all",
        help="Specify which pipeline step to run (default: all)"
    )
    args = parser.parse_args()

    if args.step == "all":
        print("Starting Complete RAG Explanation Stability Pipeline...")
        for name, script, desc in STEPS:
            run_script(script, desc)
        print("\nEntire Pipeline Completed Successfully!")
        print("Run `streamlit run src/dashboard.py` to view results interactively.")
    else:
        for name, script, desc in STEPS:
            if args.step == name:
                run_script(script, desc)
                break

if __name__ == "__main__":
    main()
