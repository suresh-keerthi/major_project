import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from typing import List
from concurrent.futures import ThreadPoolExecutor
import torch
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import time
from collections import defaultdict

# Suppress warnings
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Logging
logging.basicConfig(
    filename="evaluation_errors.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Gemini configuration
API_KEY = "AIzaSyBrZwQ6ktJDNodgj10SdaB6TqqK8V4O6so"
client = genai.Client(api_key=API_KEY)
MODEL_ID = "gemini-3-flash-preview"

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_cache = defaultdict(lambda: None)


def sanitize_text(text: str) -> str:
    replacements = {
        "persecuting": "criticizing",
        "discrimination": "bias",
        "traitors": "opponents",
        "death": "harm",
        "segregated": "divided"
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=5, max=40))
def batch_expand_ground_truth(gt_targets: List[str], contexts: List[str]):
    
    sanitized_contexts = [sanitize_text(ctx) for ctx in contexts]

    prompt = f"""
You are an expert linguistic assistant. Your task is to generate alternative expressions (synonyms or conceptually similar phrases) for given target phrases based on their specific tweet contexts.

Return ONLY a valid JSON array of arrays. Do not include markdown formatting or conversational text.

Input: A list of targets and their corresponding tweet contexts.

For each target, return exactly TWO alternative expressions that preserve the exact same meaning as the target in the context of the tweet.
- If the target is highly specific (like a proper noun), provide close variations or contextual equivalents.
- Keep the alternatives concise (1-3 words usually).

Output format:
[
 ["alt1", "alt2"],
 ["alt1", "alt2"]
]

Targets and Contexts:
"""
    for i, (gt, ctx) in enumerate(zip(gt_targets, sanitized_contexts)):
        prompt += f"{i+1}. Target: '{gt}', Context: '{ctx}'\n"

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )

        time.sleep(12)

        text = response.text.strip()

        try:
            parsed = json.loads(text)

            if not isinstance(parsed, list):
                raise ValueError("Invalid JSON format")

            expanded = []

            for i, item in enumerate(parsed[:len(gt_targets)]):
                if isinstance(item, list) and len(item) > 0:
                    expanded.append(item)
                else:
                    expanded.append([gt_targets[i]])

            # Pad with original GTs if Gemini returned fewer arrays than requested
            while len(expanded) < len(gt_targets):
                expanded.append([gt_targets[len(expanded)]])

            return expanded, True

        except Exception as e:
            msg = f"GT expansion parsing failed: {e}"
            logging.warning(msg)
            print(f"⚠️  {msg}")
            return [[gt] for gt in gt_targets], False

    except Exception as e:
        msg = f"GT expansion failed: {e}"
        logging.error(msg)
        print(f"❌  {msg}")
        return [[gt] for gt in gt_targets], False


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=5, max=40))
def batch_llm_boolean_relevance(tweets: List[str], predicted_targets: List[str]):

    prompt = """You are an expert data annotator evaluating Stance Detection targets.
Your task is to determine if a predicted target is relevant to the main subject of a given tweet.

A target is RELEVANT ("Yes") if the tweet expresses a clear stance (favor, against, or neutral) towards that specific target entity, concept, or person.
A target is IRRELEVANT ("No") if the tweet does not discuss it, or if it is merely a peripheral entity not central to the stance.

Answer EXPLICITLY and ONLY with "Yes" or "No", printing one answer per line. Do not include explanations.

Input data:
"""

    for i, (tweet, pred) in enumerate(zip(tweets, predicted_targets)):
        prompt += f"{i+1}. Tweet: '{tweet}'\nPredicted Target: '{pred}'\n"

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )

        time.sleep(12)

        answers = response.text.strip().split("\n")

        # Pad missing answers if Gemini returned fewer lines than expected
        bool_answers = ["yes" in a.lower() for a in answers]
        while len(bool_answers) < len(tweets):
            bool_answers.append(False)

        return bool_answers[:len(tweets)], True

    except Exception as e:
        msg = f"Relevance check failed: {e}"
        logging.error(msg)
        print(f"❌  {msg}")
        return [False] * len(tweets), False


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=5, max=40))
def batch_llm_likert_score(tweets, gt_targets, predicted_targets):

    prompt = """You are an expert evaluator for Stance Detection models.
Your task is to score how well a PREDICTED target matches the true GROUND TRUTH (GT) target in the context of a specific tweet.

Score from 1 to 5 using this strict rubric:
5 = Perfect match or exact synonym (e.g., GT: "climate change", Pred: "global warming").
4 = Highly relevant and captures the core meaning, but slightly broader or narrower (e.g., GT: "feminism", Pred: "women's rights").
3 = Moderately relevant, captures some aspect but misses the exact nuance (e.g., GT: "gun control", Pred: "laws").
2 = Barely relevant, tangential connection (e.g., GT: "veganism", Pred: "food").
1 = Completely incorrect, opposite, or entirely unrelated (e.g., GT: "atheism", Pred: "christianity").

Output ONLY the score numbers, one per line. No extra text, labels, or explanations.

Input data:
"""

    for i, (tweet, gt, pred) in enumerate(zip(tweets, gt_targets, predicted_targets)):
        prompt += f"\n{i+1}. Tweet: '{tweet}'\nGT: '{gt}'\nPredicted: '{pred}'\n"

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )

        time.sleep(12)

        scores = []

        for line in response.text.split("\n"):
            digits = [c for c in line if c.isdigit()]
            scores.append(int(digits[0]) if digits else 3)
            
        # Pad missing scores if Gemini returned fewer lines than expected
        while len(scores) < len(tweets):
            scores.append(3)

        return scores[:len(tweets)], True

    except Exception as e:
        msg = f"Likert scoring failed: {e}"
        logging.error(msg)
        print(f"❌  {msg}")
        return [3] * len(tweets), False


def semantic_similarity(predicted: str, expanded_gt_list: List[str]):

    if embedding_cache[predicted] is None:
        embedding_cache[predicted] = embedding_model.encode(predicted, convert_to_tensor=True)

    pred_emb = embedding_cache[predicted]

    gt_embeddings = []

    for gt in expanded_gt_list:
        if embedding_cache[gt] is None:
            embedding_cache[gt] = embedding_model.encode(gt, convert_to_tensor=True)

        gt_embeddings.append(embedding_cache[gt])

    gt_embeddings = torch.stack(gt_embeddings)

    similarities = util.pytorch_cos_sim(pred_emb, gt_embeddings)[0].cpu().numpy()

    return float(np.max(similarities))


def process_chunk(chunk, gt_col, pred_col, precomputed_expanded_gts=None):

    tweets = chunk["tweet"].tolist()
    gt_targets = chunk[gt_col].tolist()
    predicted_targets = chunk[pred_col].tolist()
    sources = chunk.get("source", pd.Series(["Unknown"] * len(tweets))).tolist()

    results = []

    if precomputed_expanded_gts is not None:
        expanded_gts = precomputed_expanded_gts
        gt_success = True
    else:
        expanded_gts, gt_success = batch_expand_ground_truth(gt_targets, tweets)

    relevances, rel_success = batch_llm_boolean_relevance(tweets, predicted_targets)

    likerts, lik_success = batch_llm_likert_score(tweets, gt_targets, predicted_targets)

    # If all three functions succeeded for this chunk, the row is marked True. Otherwise False.
    api_success = gt_success and rel_success and lik_success

    for tweet, gt, pred, source, expanded_gt, relevance, likert in zip(
            tweets, gt_targets, predicted_targets, sources, expanded_gts, relevances, likerts):

        sim = semantic_similarity(pred, expanded_gt)

        if sim >= 0.9:

            final_score = sim * 100

            results.append({
                "Source": source,
                "API Success": api_success,
                "Expanded GTs": " | ".join(expanded_gt),
                "Similarity Score": round(sim, 3),
                "LLM Relevance": True,
                "Likert Score": 5,
                "Final Score": round(final_score, 2),
                "Decision Path": "High Similarity",
                "Model Used": MODEL_ID
            })

        else:

            score = sim * 50

            if relevance:
                score += 20
                score += (likert / 5) * 30
                path = "LLM Evaluated"
            else:
                path = "Low Similarity"

            results.append({
                "Source": source,
                "API Success": api_success,
                "Expanded GTs": " | ".join(expanded_gt),
                "Similarity Score": round(sim, 3),
                "LLM Relevance": relevance,
                "Likert Score": likert,
                "Final Score": round(score, 2),
                "Decision Path": path,
                "Model Used": MODEL_ID
            })

    return results


def evaluate_batch_to_csv(input_files, summary_file, chunk_size=40,
                          gt_col="true_target", pred_col="pred_target_with_analysis",
                          reference_csv=None):

    summaries = []

    ref_df = pd.read_csv(reference_csv) if reference_csv else None

    for file in input_files:

        df = pd.read_csv(file)

        # Initialize columns to allow real-time saving and resuming
        eval_cols = ["API Success", "Expanded GTs", "Similarity Score", "LLM Relevance", 
                     "Likert Score", "Final Score", "Decision Path", "Model Used"]
        for col in eval_cols:
            if col not in df.columns:
                df[col] = None

        if "source" not in df.columns:
            df["source"] = "Unknown"

        # Only process rows that have valid string targets but haven't been scored yet
        df_valid = df.dropna(subset=["tweet", gt_col, pred_col])
        df_unscored = df_valid[df_valid["Final Score"].isna()]

        chunks = [df_unscored.iloc[i:i+chunk_size] for i in range(0, len(df_unscored), chunk_size)]

        print(f"File {file}: {len(df_unscored)} unscored rows remaining out of {len(df_valid)} valid rows.")

        for chunk in chunks:

            precomputed = None
            if ref_df is not None:
                precomputed = []
                for idx in chunk.index:
                    if idx in ref_df.index and pd.notna(ref_df.at[idx, "Expanded GTs"]):
                        precomputed.append(str(ref_df.at[idx, "Expanded GTs"]).split(" | "))
                    else:
                        precomputed.append([chunk.at[idx, gt_col]])

            chunk_results = process_chunk(chunk, gt_col, pred_col, precomputed)

            for idx, result in zip(chunk.index, chunk_results):
                df.at[idx, "source"] = result["Source"]
                df.at[idx, "API Success"] = result["API Success"]
                df.at[idx, "Expanded GTs"] = result["Expanded GTs"]
                df.at[idx, "Similarity Score"] = result["Similarity Score"]
                df.at[idx, "LLM Relevance"] = result["LLM Relevance"]
                df.at[idx, "Likert Score"] = result["Likert Score"]
                df.at[idx, "Final Score"] = result["Final Score"]
                df.at[idx, "Decision Path"] = result["Decision Path"]
                df.at[idx, "Model Used"] = result["Model Used"]

            # Save progress in real-time
            df.to_csv(file, index=False)
            logging.info(f"Saved real-time progress for a chunk in {file}")

        summaries.append({
            "File": file,
            "Avg Similarity": df["Similarity Score"].mean() if "Similarity Score" in df.columns else 0.0,
            "Avg Final Score": df["Final Score"].mean() if "Final Score" in df.columns else 0.0
        })

    pd.DataFrame(summaries).to_csv(summary_file, index=False)


if __name__ == "__main__":

    import shutil

    shutil.copy("agentic_full_results_fixed.csv", "eval_with_analysis.csv")

    evaluate_batch_to_csv(
        ["eval_with_analysis.csv"],
        "summary_with_analysis.csv",
        gt_col="true_target",
        pred_col="pred_target_with_analysis"
    )

    shutil.copy("agentic_full_results_fixed.csv", "eval_no_analysis.csv")

    evaluate_batch_to_csv(
        ["eval_no_analysis.csv"],
        "summary_no_analysis.csv",
        gt_col="true_target",
        pred_col="pred_target_no_analysis",
        reference_csv="eval_with_analysis.csv"
    )