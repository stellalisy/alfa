import json
import random

random.seed(42)


"""
generators to sample from:
1. human-written questions: /home/stellalisy/health-q/data/rank_eval/test_reference.json
2. base model generations: /home/stellalisy/health-q/data/rank_eval/model_generations/base_models/llama-3.2-3b_reference_prompt.jsonl
3. DPO policy fusion: /home/stellalisy/health-q/data/rank_eval/model_generations/dimensions/llama-3.2-3b-sft_lr5e-6_dpo_lr5e-7_beta2_average_filtered.jsonl
4. PPO reward fusion: /home/stellalisy/health-q/data/rank_eval/model_generations/ppo/llama-3.2-3b_sft_lr5e-6_ep2_rm_lr9e-6_ppo_lr5e-7_batch256_rewardfusion2_filtered.jsonl
5. PPO policy fusion: /home/stellalisy/health-q/data/rank_eval/model_generations/ppo/llama-3.2-3b_sft_lr5e-6_ep2_rm_lr9e-6_ppo_lr5e-7_batch256_policyfusion_filtered.jsonl
6. DPO coarse attributes: /home/stellalisy/health-q/data/rank_eval/model_generations/dimensions/llama-3.1-8b-sft_lr5e-6_dpo_lr5e-7_beta2_coarse_filtered_r2.jsonl
"""

def load_data_from_jsonl(file_path):
    """
    Load data from a JSONL file.
    
    Args:
        file_path (str): Path to the JSONL file.
        
    Returns:
        list: List of dictionaries loaded from the JSONL file.
    """
    data = []
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    data = {sample["id"]: sample for sample in data if sample["best_of_n"]==0}
    return data

def load_data_from_json(file_path):
    """
    Load data from a JSON file.
    SHOULD ONLY BE USED FOR HUMAN-WRITTEN QUESTIONS
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        list: List of dictionaries loaded from the JSON file.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    data = {sample["id"]: sample for sample in data}
    return data

target_generator_files = {
    "output_human": "/home/stellalisy/health-q/data/rank_eval/test_reference.json",
    "output_base": "/home/stellalisy/health-q/data/rank_eval/model_generations/base_models/llama-3.2-3b-instruct_reference_prompt.jsonl",
    "output_dpo_policy": "/home/stellalisy/health-q/data/rank_eval/model_generations/dimensions/llama-3.2-3b-sft_lr5e-6_dpo_lr5e-7_beta2_average_filtered.jsonl",
    "output_ppo_reward": "/home/stellalisy/health-q/data/rank_eval/model_generations/ppo/llama-3.2-3b_sft_lr5e-6_ep2_rm_lr9e-6_ppo_lr5e-7_batch256_rewardfusion2_filtered.jsonl",
    "output_ppo_policy": "/home/stellalisy/health-q/data/rank_eval/model_generations/ppo/llama-3.2-3b_sft_lr5e-6_ep2_rm_lr9e-6_ppo_lr5e-7_batch256_policyfusion_filtered.jsonl",
    "output_dpo_coarse": "/home/stellalisy/health-q/data/rank_eval/model_generations/dimensions/llama-3.1-8b-sft_lr5e-6_dpo_lr5e-7_beta2_coarse_filtered_r2.jsonl",
}

human_data = load_data_from_json(target_generator_files["output_human"])

random_ids = random.sample(list(human_data.keys()), 100)

contxets = {i: human_data[i]["instruction"] for i in random_ids}
human_outputs = {i: human_data[i]["output_human"] for i in random_ids}

generators_outputs = {}
for generator, filepath in target_generator_files.items():
    if generator == "output_human":
        continue
    data = load_data_from_jsonl(filepath)
    
    generator_outputs = {i: data[i]["output"] for i in random_ids}
    generators_outputs[generator] = generator_outputs

otuputs = []
for i in random_ids:
    output = {
        "id": i,
        "context": contxets[i],
        "human_output": human_outputs[i],
    }
    for generator, generator_outputs in generators_outputs.items():
        output[generator] = generator_outputs[i]
    
    otuputs.append(output)

with open("/home/stellalisy/health-q/data/rank_eval/sample_for_human_eval.json", "w") as f:
    json.dump(otuputs, f, indent=4)