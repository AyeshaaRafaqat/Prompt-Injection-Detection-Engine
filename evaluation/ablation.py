"""
PIDE Ablation Study Script
Analyzes the contribution of each layer to the overall performance.
"""

import os
import logging
import pandas as pd
from datasets import load_dataset
from pipeline import load_pipeline, detect
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PIDE.Ablation")

def run_ablation():
    """Runs ablation study across 4 configurations."""
    os.makedirs("evaluation/results", exist_ok=True)
    
    logger.info("Loading test dataset...")
    dataset = load_dataset("deepset/prompt-injections", split="test")
    layers = load_pipeline()
    l1, l2, l3, l4 = layers
    
    configs = [
        {"name": "L1 Only", "mask": (True, False, False), "notes": "Regex baseline"},
        {"name": "L1 + L2", "mask": (True, True, False), "notes": "Regex + Semantic"},
        {"name": "L1 + L2 + L3", "mask": (True, True, True), "notes": "All layers, equal weights"},
        {"name": "Full System (L4)", "mask": (True, True, True), "notes": "Optimized L4 weights"}
    ]
    
    results = []
    
    for config in configs:
        logger.info(f"Running config: {config['name']}...")
        y_true = []
        y_pred = []
        
        for row in dataset:
            prompt = row['text']
            label = row['label']
            
            # Layer executions
            s1 = l1.score(prompt)[0] if config['mask'][0] else 0.0
            s2 = l2.score(prompt)[0] if config['mask'][1] else 0.0
            s3 = l3.score(prompt)[0] if config['mask'][2] else 0.0
            
            # Decision logic
            if config['name'] == "Full System (L4)":
                risk, decision = l4.score(s1, s2, s3)
            elif config['name'] == "L1 + L2 + L3":
                risk = (s1 + s2 + s3) / 3.0
                decision = "BLOCK" if risk >= 0.5 else "ALLOW"
            else:
                # Sum of active layers
                risk = s1 + s2 + s3
                decision = "BLOCK" if risk >= 0.5 else "ALLOW"
                
            pred = 1 if decision in ["BLOCK", "SANITISE"] else 0
            y_true.append(label)
            y_pred.append(pred)
            
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        results.append({
            "config": config['name'],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "fpr": round(fpr, 4),
            "notes": config['notes']
        })

    df = pd.DataFrame(results)
    logger.info("\nAblation Results:\n" + df.to_string(index=False))
    df.to_csv("evaluation/results/ablation_table.csv", index=False)
    logger.info("Ablation study complete.")

if __name__ == "__main__":
    run_ablation()
