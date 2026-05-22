"""
PIDE Evaluation Script
Computes performance metrics on the deepset/prompt-injections test split.
"""

import os
import logging
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from datasets import load_dataset
from pipeline import load_pipeline, detect

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PIDE.Eval")

def evaluate():
    """Runs evaluation on the test split and saves results."""
    # Create results directory
    os.makedirs("evaluation/results", exist_ok=True)
    
    logger.info("Loading test dataset...")
    dataset = load_dataset("deepset/prompt-injections", split="test")
    
    logger.info("Initializing pipeline...")
    layers = load_pipeline()
    
    y_true = []
    y_pred = []
    
    logger.info(f"Evaluating {len(dataset)} samples...")
    
    for i, row in enumerate(dataset):
        prompt = row['text']
        label = row['label'] # 1 for injection, 0 for benign
        
        res = detect(prompt, *layers)
        
        # Binary prediction: BLOCK or SANITISE -> 1, ALLOW -> 0
        pred = 1 if res['decision'] in ["BLOCK", "SANITISE"] else 0
        
        y_true.append(label)
        y_pred.append(pred)
        
        if (i+1) % 50 == 0:
            logger.info(f"Processed {i+1}/{len(dataset)} samples...")

    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    # False Positive Rate
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    report = (
        "PIDE Evaluation Report\n"
        "======================\n"
        f"Samples: {len(dataset)}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall:    {recall:.4f}\n"
        f"F1 Score:  {f1:.4f}\n"
        f"FPR:       {fpr:.4f}\n"
        "======================\n"
    )
    
    logger.info("\n" + report)
    
    # Save text report
    with open("evaluation/results/full_system_report.txt", "w") as f:
        f.write(report)
        
    # Save Confusion Matrix plot
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Injection"])
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("PIDE Confusion Matrix")
    plt.savefig("evaluation/results/confusion_matrix.png")
    plt.close()
    
    logger.info("Evaluation complete. Results saved to evaluation/results/")

if __name__ == "__main__":
    evaluate()
