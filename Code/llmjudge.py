import torch
import re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
MODEL_ID = "openai/gpt-oss-20b"
RESULTS_FILE = "fixedresults.txt"
RUBRIC_FILE = "Criterion.md" 

print("Loading GPT-OSS-20B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto", 
    device_map="auto",
    trust_remote_code=True
)
def parse_results(file_path):
    print(f"Reading {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    if not content:
        print("ERROR: fixedresults.txt is empty!")
        return []

    entries = content.split("============================================================")
    print(f"Split file into {len(entries)} raw segments.")
    
    data = []
    for i, entry in enumerate(entries):
        if not entry.strip(): continue
        
        #regex to catch the content
        user_match = re.search(r"user\s*(.*?)\s*<\|im_start\|>assistant", entry, re.DOTALL)
        baseline_match = re.search(r"BASELINE OUTPUT:\s*(.*?)\s*------------------------------", entry, re.DOTALL)
        # Captures until the next separator or end of string
        steered_match = re.search(r"STEERED OUTPUT:\s*(.*)", entry, re.DOTALL)
        
        if user_match and baseline_match and steered_match:
            data.append({
                "query": user_match.group(1).strip(),
                "baseline": baseline_match.group(1).strip(),
                "steered": steered_match.group(1).strip()
            })
        else:
            print(f"Warning: Segment {i} failed to match regex patterns.")

    return data

# 3. JUDGING FUNCTION
def evaluate(rubric, query, response, label):
    judge_prompt = f"{rubric}\n\n[DATA TO EVALUATE]\nUser Query: {query}\nAssistant ({label}): {response}"
    
    messages = [
        {"role": "system", "content": "You are a professional pedagogical auditor. Use medium reasoning effort. Grade the response strictly according to the rubric."},
        {"role": "user", "content": judge_prompt}
    ]

    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", reasoning_effort="medium").to(model.device)
    attention_mask = torch.ones(inputs.shape, device=model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs, 
            attention_mask=attention_mask, 
            max_new_tokens=1000, 
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

# 4. SUMMARY LOGIC
def calculate_summary(results_list):
    stats = {
        "BASELINE": {"Pedagogy": [], "Relevance": [], "NoSpoilers": 0, "Brevity": 0},
        "STEERED": {"Pedagogy": [], "Relevance": [], "NoSpoilers": 0, "Brevity": 0}
    }

    for res in results_list:
        for mode in ["BASELINE", "STEERED"]:
            text = res[f"{mode.lower()}_grade"]
            p_match = re.search(r"Criterion 3.*?Rating:\s*(\d)", text, re.DOTALL)
            r_match = re.search(r"Criterion 4.*?Rating:\s*(\d)", text, re.DOTALL)
            s_match = re.search(r"Criterion 1.*?Refusal presence:\s*(Yes|No)", text, re.DOTALL | re.IGNORECASE)
            b_match = re.search(r"Criterion 2.*?Requirement met:\s*(Yes|No)", text, re.DOTALL | re.IGNORECASE)

            if p_match: stats[mode]["Pedagogy"].append(int(p_match.group(1)))
            if r_match: stats[mode]["Relevance"].append(int(r_match.group(1)))
            if s_match and "yes" in s_match.group(1).lower(): stats[mode]["NoSpoilers"] += 1
            if b_match and "yes" in b_match.group(1).lower(): stats[mode]["Brevity"] += 1

    summary = "\n" + "="*50 + "\n            FINAL EVALUATION SUMMARY\n" + "="*50 + "\n"
    for mode in ["BASELINE", "STEERED"]:
        avg_p = np.mean(stats[mode]["Pedagogy"]) if stats[mode]["Pedagogy"] else 0
        avg_r = np.mean(stats[mode]["Relevance"]) if stats[mode]["Relevance"] else 0
        summary += f"\n--- {mode} MODEL ---\n"
        summary += f"No Spoilers (C1 Yes):  {stats[mode]['NoSpoilers']} / {len(results_list)}\n"
        summary += f"Brevity Met (C2 Yes):  {stats[mode]['Brevity']} / {len(results_list)}\n"
        summary += f"Pedagogical Avg (C3): {avg_p:.2f} / 5\n"
        summary += f"Relevance Avg (C4):   {avg_r:.2f} / 5\n"
    return summary

try:
    with open(RUBRIC_FILE, "r") as f:
        rubric_text = f.read()
except FileNotFoundError:
    print(f"ERROR: Could not find rubric file: {RUBRIC_FILE}")
    exit()

test_cases = parse_results(RESULTS_FILE)
print(f"Final Count: Found {len(test_cases)} valid test cases to judge.")

if len(test_cases) == 0:
    print("Stopping: No cases found to process. Check the regex in parse_results.")
    exit()

all_results = []
with open("final_evaluation_report.txt", "w") as f:
    for i, case in enumerate(test_cases):
        print(f"Grading Prompt {i+1}/{len(test_cases)}...")
        b_grade = evaluate(rubric_text, case['query'], case['baseline'], "BASELINE")
        s_grade = evaluate(rubric_text, case['query'], case['steered'], "STEERED")
        
        all_results.append({"baseline_grade": b_grade, "steered_grade": s_grade})
        
        f.write(f"PROMPT {i+1}: {case['query'][:100]}...\n")
        f.write(f"\n--- BASELINE EVALUATION ---\n{b_grade}\n")
        f.write(f"\n--- STEERED EVALUATION ---\n{s_grade}\n")
        f.write(f"\n{'='*80}\n")
        f.flush()

summary_report = calculate_summary(all_results)
print(summary_report)
with open("final_evaluation_report.txt", "a") as f:
    f.write(summary_report)