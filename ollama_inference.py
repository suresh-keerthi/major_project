import pandas as pd
import requests
import csv
import re
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "tweet-stance"


# -------------------------
# CLEAN GENERATED TEXT (CRITICAL)
# -------------------------
def clean_generated_text(text):
    if not isinstance(text, str):
        return ""

    # Remove prompt leakage
    text = text.split("###")[0]

    # Remove keywords if leaked
    text = re.sub(r"(Instruction|Input|Response):?", "", text, flags=re.IGNORECASE)

    # Remove quotes
    text = text.replace('"', '').replace("'", "")

    return text.strip()


# -------------------------
# CORE GENERATION
# -------------------------
def generate(prompt, max_tokens=20):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.0
            }
        },
        timeout=300
    )
    return response.json()["response"].strip()


# -------------------------
# STAGE 1: ANALYSIS
# -------------------------
def analyze(tweet):
    prompt = f"""### Instruction:
Briefly describe sentiment and tone in one short line.

### Input:
Tweet: {tweet}

### Response:
"""
    out = generate(prompt, max_tokens=30)
    return clean_generated_text(out)


# -------------------------
# TARGET (NO ANALYSIS)
# -------------------------
def predict_target(tweet):
    prompt = f"""### Instruction:
Identify the primary target (topic) of the following tweet. Output only the target name.

### Input:
Tweet: {tweet}

### Response:
"""
    out = generate(prompt, max_tokens=10)
    out = clean_generated_text(out)
    return ' '.join(out.split()[:3])


# -------------------------
# TARGET (WITH ANALYSIS)
# -------------------------
def predict_target_with_analysis(tweet, analysis):
    prompt = f"""### Instruction:
Identify the primary target (topic) of the following tweet. Output only the target name.

### Input:
Tweet: {tweet}
Additional context : {analysis}

### Response:
"""
    out = generate(prompt, max_tokens=10)
    out = clean_generated_text(out)
    return ' '.join(out.split()[:3])


# -------------------------
# STANCE (NO ANALYSIS)
# -------------------------
def predict_stance(tweet, target):
    prompt = f"""### Instruction:
Determine the stance (FAVOR, AGAINST, or NONE) towards the provided target.

### Input:
Target: {target}
Tweet: {tweet}

### Response:
"""
    out = generate(prompt, max_tokens=5)
    out = clean_generated_text(out).upper()

    if "FAVOR" in out:
        return "FAVOR"
    elif "AGAINST" in out:
        return "AGAINST"
    else:
        return "NONE"


# -------------------------
# STANCE (WITH ANALYSIS)
# -------------------------
def predict_stance_with_analysis(tweet, target, analysis):
    prompt = f"""### Instruction:
Determine the stance (FAVOR, AGAINST, or NONE) towards the provided target.

### Input:
Target: {target}
Tweet: {tweet}
Additional context : {analysis}

### Response:
"""
    out = generate(prompt, max_tokens=5)
    out = clean_generated_text(out).upper()

    if "FAVOR" in out:
        return "FAVOR"
    elif "AGAINST" in out:
        return "AGAINST"
    else:
        return "NONE"


# -------------------------
# MAIN PIPELINE
# -------------------------
def run(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "tweet",
            "true_target",
            "true_stance",
            "source",
            "analysis",
            "pred_target_no_analysis",
            "pred_stance_no_analysis",
            "pred_target_with_analysis",
            "pred_stance_with_analysis"
        ])

        for i, row in tqdm(df.iterrows(), total=len(df)):
            tweet = row["tweet"]  # ✅ no cleaning

            try:
                # ---- NO ANALYSIS ----
                target = predict_target(tweet)
                stance = predict_stance(tweet, target)

                # ---- WITH ANALYSIS ----
                analysis = analyze(tweet)
                target_a = predict_target_with_analysis(tweet, analysis)
                stance_a = predict_stance_with_analysis(tweet, target_a, analysis)

            except Exception as e:
                print(f"Error at row {i}:", e)
                analysis = ""
                target, stance = "", "NONE"
                target_a, stance_a = "", "NONE"

            writer.writerow([
                tweet,
                row.get("target", ""),
                row.get("stance", ""),
                row.get("source", ""),
                analysis,
                target,
                stance,
                target_a,
                stance_a
            ])

            f.flush()

            if i % 100 == 0:
                print(f"Processed {i}/{len(df)}")


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    run(
        "C:\\Users\\sures\\inference\\merged_test_set.csv",
        "agentic_full_results_fixed.csv"
    )