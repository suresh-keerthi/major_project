import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bertscore


# ------------------------------
# LOAD DATA
# ------------------------------
df = pd.read_csv("agentic_full_results_fixed.csv")

sources = df["source"].unique()

rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
smooth = SmoothingFunction().method1


# ------------------------------
# TEXT CLEANING
# ------------------------------
def clean_text(x):
    return str(x).lower().strip()


# ------------------------------
# TARGET METRIC CALCULATION
# ------------------------------
def compute_target_metrics(true, pred):

    true = clean_text(true)
    pred = clean_text(pred)

    # BLEU
    bleu = sentence_bleu(
        [true.split()],
        pred.split(),
        smoothing_function=smooth
    )

    # ROUGE-L
    rouge_l = rouge.score(true, pred)['rougeL'].fmeasure

    return bleu, rouge_l


# ------------------------------
# RECALL BASED ON THRESHOLDS
# ------------------------------
def compute_recall(bert, bleu, rouge_l):

    if (
        bert >= 0.7 and
        bleu >= 0.2 and
        rouge_l >= 0.4
    ):
        return 1
    else:
        return 0


# ------------------------------
# C-SCORE
# ------------------------------
def compute_cscore(bert, bleu, rouge_l, recall):

    similarity = (0.6 * bert + 0.2 * bleu + 0.2 * rouge_l)

    return similarity * recall


# ------------------------------
# BERTSCORE FUNCTION
# ------------------------------
def compute_bertscore(true_list, pred_list):

    P, R, F = bertscore(
        pred_list,
        true_list,
        lang="en",
        verbose=False
    )

    return F.tolist()


# ------------------------------
# MAIN EVALUATION
# ------------------------------
results = []

for src in sources:

    subset = df[df["source"] == src]

    # ----------------------
    # STANCE METRICS
    # ----------------------

    y_true = subset["true_stance"].str.upper()

    pred_no = subset["pred_stance_no_analysis"].str.upper()
    pred_with = subset["pred_stance_with_analysis"].str.upper()

    stance_acc_no = accuracy_score(y_true, pred_no)
    stance_acc_with = accuracy_score(y_true, pred_with)

    f1_macro_no = f1_score(y_true, pred_no, average="macro")
    f1_macro_with = f1_score(y_true, pred_with, average="macro")

    f1_weight_no = f1_score(y_true, pred_no, average="weighted")
    f1_weight_with = f1_score(y_true, pred_with, average="weighted")


    # ----------------------
    # TARGET METRICS
    # ----------------------

    true_targets = subset["true_target"].fillna("").tolist()

    pred_targets_no = subset["pred_target_no_analysis"].fillna("").tolist()
    pred_targets_with = subset["pred_target_with_analysis"].fillna("").tolist()


    # BERTScore
    bert_no_list = compute_bertscore(true_targets, pred_targets_no)
    bert_with_list = compute_bertscore(true_targets, pred_targets_with)


    bleu_no = []
    rouge_no = []
    recall_no = []
    cscore_no_list = []

    bleu_with = []
    rouge_with = []
    recall_with = []
    cscore_with_list = []


    # ----------------------
    # LOOP OVER SAMPLES
    # ----------------------

    for t, p1, p2, b1, b2 in zip(
        true_targets,
        pred_targets_no,
        pred_targets_with,
        bert_no_list,
        bert_with_list
    ):

        # -------- NO ANALYSIS --------

        bleu1, rouge1 = compute_target_metrics(t, p1)

        rec1 = compute_recall(b1, bleu1, rouge1)

        cscore1 = compute_cscore(b1, bleu1, rouge1, rec1)

        bleu_no.append(bleu1)
        rouge_no.append(rouge1)
        recall_no.append(rec1)
        cscore_no_list.append(cscore1)


        # -------- WITH ANALYSIS --------

        bleu2, rouge2 = compute_target_metrics(t, p2)

        rec2 = compute_recall(b2, bleu2, rouge2)

        cscore2 = compute_cscore(b2, bleu2, rouge2, rec2)

        bleu_with.append(bleu2)
        rouge_with.append(rouge2)
        recall_with.append(rec2)
        cscore_with_list.append(cscore2)


    # ----------------------
    # AVERAGES
    # ----------------------

    results.append({

        "dataset": src,

        # stance metrics
        "accuracy_no_analysis": stance_acc_no,
        "accuracy_with_analysis": stance_acc_with,

        "f1_macro_no_analysis": f1_macro_no,
        "f1_macro_with_analysis": f1_macro_with,

        "f1_weighted_no_analysis": f1_weight_no,
        "f1_weighted_with_analysis": f1_weight_with,

        # target metrics
        "bert_no_analysis": np.mean(bert_no_list),
        "bert_with_analysis": np.mean(bert_with_list),

        "bleu_no_analysis": np.mean(bleu_no),
        "bleu_with_analysis": np.mean(bleu_with),

        "rougeL_no_analysis": np.mean(rouge_no),
        "rougeL_with_analysis": np.mean(rouge_with),

        "recall_no_analysis": np.mean(recall_no),
        "recall_with_analysis": np.mean(recall_with),

        "cscore_no_analysis": np.mean(cscore_no_list),
        "cscore_with_analysis": np.mean(cscore_with_list)
    })


# ------------------------------
# SAVE RESULTS
# ------------------------------

results_df = pd.DataFrame(results)

print("\nEvaluation Results:\n")
print(results_df)

results_df.to_csv("evaluation_results_per_dataset.csv", index=False)

print("\nSaved to evaluation_results_per_dataset.csv")