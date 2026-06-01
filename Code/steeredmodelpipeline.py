from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest
import os
import re

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Initialize the LLM model
llm = LLM(model="Qwen/Qwen3-14B", enable_steer_vector=True, enforce_eager=True, tensor_parallel_size=1, enable_chunked_prefill=False, enable_prefix_caching=False, max_model_len=4096,)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=4096,
)
text = """<|im_start|>system
You are a strict Socratic tutor. 
1. BREVITY: Keep internal reasoning to 15 words. 
2. NO SPOILERS: Never reveal the final answer or solution.
<|im_end|>
<|im_start|>user
Janet's grades for her first semester of college were 90, 80, 70, and 100. If her semester 2 average was 82 percent, how much higher was her semester 1 average compared to her semester 2 average?
<|im_start|>assistant
"""
target_layers = list(range(10,32))

baseline_request = SteerVectorRequest("baseline", 1, steer_vector_local_path="vectors/qwen_14b_socratic.gguf", scale=0, target_layers=target_layers, prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])
baseline_output = llm.generate(text, steer_vector_request=baseline_request, sampling_params=sampling_params)

socratic_request = SteerVectorRequest("socratic", 2, steer_vector_local_path="vectors/qwen_14b_socratic.gguf", scale=2, target_layers=target_layers, prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])
socratic_output = llm.generate(text, steer_vector_request=socratic_request, sampling_params=sampling_params)
#print(text)
#print("---")
#print("Baseline output")
#print(baseline_output[0].outputs[0].text)
#print("---")
#print("Steered output")
#print(socratic_output[0].outputs[0].text)

def clean_output(text):
    # This removes everything between <think> and </think> inclusive
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

# inside write logic
with open("fixedresults.txt", "a", encoding="utf-8") as f_clean, \
     open("fixedresultswthink.txt", "a", encoding="utf-8") as f_full:
    
    header = f"PROMPT: {text}\n" + ("-" * 30) + "\n"
    separator = "-" * 30 + "\n"
    footer = "=" * 60 + "\n\n"

    for f in [f_clean, f_full]:
        f.write(header)

    # --- BASELINE ---
    raw_baseline = baseline_output[0].outputs[0].text
    
    f_clean.write("BASELINE OUTPUT:\n")
    f_clean.write(clean_output(raw_baseline) + "\n")
    
    f_full.write("BASELINE OUTPUT (RAW):\n")
    f_full.write(raw_baseline + "\n")

    for f in [f_clean, f_full]:
        f.write(separator)

    # --- STEERED ---
    raw_steered = socratic_output[0].outputs[0].text
    
    f_clean.write("STEERED OUTPUT:\n")
    f_clean.write(clean_output(raw_steered) + "\n")
    
    f_full.write("STEERED OUTPUT (RAW):\n")
    f_full.write(raw_steered + "\n")

    for f in [f_clean, f_full]:
        f.write(footer)