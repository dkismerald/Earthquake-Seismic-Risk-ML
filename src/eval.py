# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
from src.logger import get_logger

logger = get_logger("eval")

def plot_roc(y_true, y_prob, out_png="roc.png"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    score = auc(fpr, tpr)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"AUC = {score:.3f}")
    plt.plot([0,1],[0,1], linestyle='--', color='grey')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    logger.info("Saved ROC plot: %s", out_png)

def plot_pr(y_true, y_prob, out_png="pr.png"):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(6,4))
    plt.plot(r, p, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    logger.info("Saved PR plot: %s", out_png)
