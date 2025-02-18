# Usage:  huggingface-cli upload [repo_id] [local_path] [path_in_repo]

import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

def average_model_weights(model_dirs, output_dir, save_average=True, missing_dims=[]):
    """
    Averages the weights of multiple models and saves the resulting model.

    Args:
        model_dirs (list): List of directories containing the trained models.
        output_dir (str): Directory to save the averaged model.
    """
    # Load all models into a list
    models = [AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16) for model_dir in model_dirs]

    if save_average:
        # Get the state_dict of the first model as a template
        averaged_state_dict = copy.deepcopy(models[0].state_dict())

        # Initialize the averaged state dict with zeros
        for key in averaged_state_dict:
            averaged_state_dict[key] = torch.zeros_like(averaged_state_dict[key])

        # Sum the weights of all models
        for i, model in enumerate(models):
            for key in model.state_dict():
                averaged_state_dict[key] += model.state_dict()[key]
            print(f"Loaded model {i+1}/{len(models)}")

        # Divide by the number of models to get the average
        num_models = len(models)
        for key in averaged_state_dict:
            averaged_state_dict[key] /= num_models

        # Save the averaged model
        averaged_model = AutoModelForCausalLM.from_pretrained(model_dirs[0])  # Use the first model's config
        averaged_model.load_state_dict(averaged_state_dict)
        averaged_model.save_pretrained(output_dir)

        tokenizer = AutoTokenizer.from_pretrained(model_dirs[0])
        # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        tokenizer.save_pretrained(output_dir)
        print(f"saved {output_dir}")

    for missing_dim in missing_dims:
        missing_idx = [missing_dim in model_dir for model_dir in model_dirs]
        missing_idx = missing_idx.index(True)
        output_dir = model_dirs[missing_idx].replace(missing_dim, f"wo{missing_dim}")
        models_missing_one = models[:missing_idx] + models[missing_idx+1:]
        averaged_state_dict_missing_one = copy.deepcopy(models_missing_one[0].state_dict())
        for key in averaged_state_dict_missing_one:
            averaged_state_dict_missing_one[key] = torch.zeros_like(averaged_state_dict_missing_one[key])
        for i, model in enumerate(models_missing_one):
            for key in model.state_dict():
                averaged_state_dict_missing_one[key] += model.state_dict()[key]
            print(f"Loaded model {i+1}/{len(models_missing_one)}")
        num_models = len(models_missing_one)
        for key in averaged_state_dict_missing_one:
            averaged_state_dict_missing_one[key] /= num_models
        averaged_model_missing_one = AutoModelForCausalLM.from_pretrained(model_dirs[0])
        averaged_model_missing_one.load_state_dict(averaged_state_dict_missing_one)
        averaged_model_missing_one.save_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        tokenizer.save_pretrained(output_dir)
        print(f"saved {output_dir}")



def average_reward_model_weights(model_dirs, output_dir):
    """
    Averages the weights of multiple models and saves the resulting model.

    Args:
        model_dirs (list): List of directories containing the trained models.
        output_dir (str): Directory to save the averaged model.
    """
    # Load all models into a list
    models = [AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=1, trust_remote_code=True, torch_dtype=torch.bfloat16) for model_dir in model_dirs]

    # Get the state_dict of the first model as a template
    averaged_state_dict = copy.deepcopy(models[0].state_dict())

    # Initialize the averaged state dict with zeros
    for key in averaged_state_dict:
        averaged_state_dict[key] = torch.zeros_like(averaged_state_dict[key])

    # Sum the weights of all models
    for i, model in enumerate(models):
        print(f"Loading model {i+1}/{len(models)}")
        for key in model.state_dict():
            averaged_state_dict[key] += model.state_dict()[key]
        print(f"Loaded model {i+1}/{len(models)}")

    # Divide by the number of models to get the average
    num_models = len(models)
    for key in averaged_state_dict:
        averaged_state_dict[key] /= num_models

    # Save the averaged model
    averaged_model = AutoModelForSequenceClassification.from_pretrained(model_dirs[0], num_labels=1)  # Use the first model's config
    averaged_model.load_state_dict(averaged_state_dict)
    averaged_model.save_pretrained(output_dir)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dirs[0])
    except:
        if "3b" in model_dirs[0]: tokenizer_path = "meta-llama/Llama-3.2-3B-Instruct"
        else: tokenizer_path = "meta-llama/Llama-3.1-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(output_dir)

# Example usage
# model_directories = [
#     "/gscratch/stf/stelli/health-q/sft_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_accuracy",
#     "/gscratch/stf/stelli/health-q/sft_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_answerability",
#     "/gscratch/stf/stelli/health-q/sft_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_avoidbias",
#     "/gscratch/stf/stelli/health-q/sft_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_clarity",
#     "/gscratch/stf/stelli/health-q/sft_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_focus",
#     "/gscratch/stf/stelli/health-q/sft_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_relevance"
# ]
# output_directory = "/gscratch/stf/stelli/health-q/sft_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_average"

# model_directories = [
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.1-8b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_accuracy_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.1-8b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_answerability_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.1-8b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_avoidbias_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.1-8b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_clarity_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.1-8b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_focus_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.1-8b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_relevance_filtered"
# ]

# output_directory = "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.1-8b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_average_filtered"

# average_model_weights(model_directories, output_directory)


# model_directories = [
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_accuracy_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_answerability_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_avoidbias_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_clarity_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_focus_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_relevance_filtered"
# ]

# output_directory = "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_average_filtered"



# model_directories = [
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.1-8b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_accuracy_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.1-8b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_answerability_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.1-8b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_avoidbias_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.1-8b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_clarity_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.1-8b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_focus_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.1-8b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_relevance_filtered"
# ]

# output_directory = "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.1-8b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_average_filtered"
# average_model_weights(model_directories, output_directory, save_average=False, missing_dims=["accuracy", "answerability", "avoidbias", "clarity", "focus", "relevance"])


# model_directories = [
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_accuracy_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_answerability_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_avoidbias_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_clarity_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_focus_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_relevance_filtered"
# ]

# output_directory = "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_average_filtered"
# average_model_weights(model_directories, output_directory, save_average=False, missing_dims=["accuracy", "answerability", "avoidbias", "clarity", "focus", "relevance"])


# model_directories = [
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_accuracy_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_answerability_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_avoidbias_filtered",
# ]

# output_directory = "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_medical_filtered"
# average_model_weights(model_directories, output_directory, save_average=True)

# model_directories = [
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_clarity_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_focus_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_relevance_filtered"
# ]

# output_directory = "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_general_filtered"
# average_model_weights(model_directories, output_directory, save_average=True)


# model_directories = [
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_accuracy_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_avoidbias_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_relevance_filtered"
# ]

# output_directory = "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_medical2_filtered"
# average_model_weights(model_directories, output_directory, save_average=True)

# model_directories = [
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_clarity_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_answerability_filtered",
#     "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_focus_filtered",
# ]

# output_directory = "/fsx-onellm/stellalisy/health-q/dpo_models/llama-3.2-3b-instruct_lr5e-6_ep2/dpo_lr5e-7_beta2_general2_filtered"
# average_model_weights(model_directories, output_directory, save_average=True)









# average_model_weights(model_directories, output_directory)

# model_directories = [
#     "jiminmun/llama-3.2-3b_reward_model_accuracy_lr9e-6_no_sys_msg_filtered",
#     "jiminmun/llama-3.2-3b_reward_model_answerability_lr9e-6_no_sys_msg_filtered",
#     "jiminmun/llama-3.2-3b_reward_model_avoidbias_lr9e-6_no_sys_msg_filtered",
#     "jiminmun/llama-3.2-3b_reward_model_clarity_lr9e-6_no_sys_msg_filtered",
#     "jiminmun/llama-3.2-3b_reward_model_focus_lr9e-6_no_sys_msg_filtered",
#     "jiminmun/llama-3.2-3b_reward_model_relevance_lr9e-6_no_sys_msg_filtered"
# ]
# output_directory = "/fsx-onellm/stellalisy/health-q/reward_models/llama-3.2-3b_reward_model_average_lr9e-6_no_sys_msg_filtered"

# average_reward_model_weights(model_directories, output_directory)


# model_directories = [
#     "/fsx-onellm/stellalisy/health-q/reward_models/llama-3.1-8b-rm_lr9e-6_accuracy_filtered",
#     "/fsx-onellm/stellalisy/health-q/reward_models/llama-3.1-8b-rm_lr9e-6_answerability_filtered",
#     "/fsx-onellm/stellalisy/health-q/reward_models/llama-3.1-8b-rm_lr9e-6_avoidbias_filtered",
#     "/fsx-onellm/stellalisy/health-q/reward_models/llama-3.1-8b-rm_lr9e-6_clarity_filtered",
#     "/fsx-onellm/stellalisy/health-q/reward_models/llama-3.1-8b-rm_lr9e-6_focus_filtered",
#     "/fsx-onellm/stellalisy/health-q/reward_models/llama-3.1-8b-rm_lr9e-6_relevance_filtered"
# ]
# output_directory = "/fsx-onellm/stellalisy/health-q/reward_models/llama-3.1-8b-rm_lr9e-6_average_filtered"
# average_reward_model_weights(model_directories, output_directory)


# model_directories = [
#     "/fsx-onellm/stellalisy/health-q/reward_models/llama-3.2-3b-rm_lr1e-2_accuracy_filtered",
#     "/fsx-onellm/stellalisy/health-q/reward_models/llama-3.2-3b-rm_lr1e-2_answerability_filtered",
#     "/fsx-onellm/stellalisy/health-q/reward_models/llama-3.2-3b-rm_lr1e-2_avoidbias_filtered",
#     "/fsx-onellm/stellalisy/health-q/reward_models/llama-3.2-3b-rm_lr1e-2_clarity_filtered",
#     "/fsx-onellm/stellalisy/health-q/reward_models/llama-3.2-3b-rm_lr1e-2_focus_filtered",
#     "/fsx-onellm/stellalisy/health-q/reward_models/llama-3.2-3b-rm_lr1e-2_relevance_filtered"
# ]
# output_directory = "/fsx-onellm/stellalisy/health-q/reward_models/llama-3.2-3b-rm_lr1e-2_average_filtered"
# average_reward_model_weights(model_directories, output_directory)

# model_directories = [
#     "/fsx-onellm/stellalisy/health-q/reward_models/llama-3.2-3b-rm_lr1e-4_accuracy_filtered_batch32",
#     "/fsx-onellm/stellalisy/health-q/reward_models/llama-3.2-3b-rm_lr1e-4_answerability_filtered_batch32",
#     "/fsx-onellm/stellalisy/health-q/reward_models/llama-3.2-3b-rm_lr1e-4_avoidbias_filtered_batch32",
#     "/fsx-onellm/stellalisy/health-q/reward_models/llama-3.2-3b-rm_lr1e-4_clarity_filtered_batch32",
#     "/fsx-onellm/stellalisy/health-q/reward_models/llama-3.2-3b-rm_lr1e-4_focus_filtered_batch32",
#     "/fsx-onellm/stellalisy/health-q/reward_models/llama-3.2-3b-rm_lr1e-4_relevance_filtered_batch32"
# ]
# output_directory = "/fsx-onellm/stellalisy/health-q/reward_models/llama-3.2-3b-rm_lr1e-4_average_filtered_batch32"
# average_reward_model_weights(model_directories, output_directory)



# model_directories = [
#     "/fsx-onellm/stellalisy/health-q/ppo_models/llama-3.2-3b-instruct_lr5e-6_ep2/rm_lr1e-4_ppo_lr5e-7_batch16_accuracy_filtered",
#     "/fsx-onellm/stellalisy/health-q/ppo_models/llama-3.2-3b-instruct_lr5e-6_ep2/rm_lr1e-4_ppo_lr5e-7_batch16_answerability_filtered",
#     "/fsx-onellm/stellalisy/health-q/ppo_models/llama-3.2-3b-instruct_lr5e-6_ep2/rm_lr1e-4_ppo_lr5e-7_batch16_avoidbias_filtered",
#     "/fsx-onellm/stellalisy/health-q/ppo_models/llama-3.2-3b-instruct_lr5e-6_ep2/rm_lr1e-4_ppo_lr5e-7_batch16_clarity_filtered",
#     "/fsx-onellm/stellalisy/health-q/ppo_models/llama-3.2-3b-instruct_lr5e-6_ep2/rm_lr1e-4_ppo_lr5e-7_batch16_focus_filtered",
#     "/fsx-onellm/stellalisy/health-q/ppo_models/llama-3.2-3b-instruct_lr5e-6_ep2/rm_lr1e-4_ppo_lr5e-7_batch16_relevance_filtered"
# ]
# output_directory = "/fsx-onellm/stellalisy/health-q/ppo_models/llama-3.2-3b-instruct_lr5e-6_ep2/rm_lr1e-4_ppo_lr5e-7_batch16_dimensions_filtered_average"
# average_model_weights(model_directories, output_directory, save_average=True)


model_directories = [
    "/fsx-onellm/stellalisy/health-q/ppo_models/llama-3.2-3b-instruct_lr5e-6_ep2/rm_lr9e-6_ppo_lr5e-7_batch256_accuracy_filtered",
    "/fsx-onellm/stellalisy/health-q/ppo_models/llama-3.2-3b-instruct_lr5e-6_ep2/rm_lr9e-6_ppo_lr5e-7_batch256_answerability_filtered",
    "/fsx-onellm/stellalisy/health-q/ppo_models/llama-3.2-3b-instruct_lr5e-6_ep2/rm_lr9e-6_ppo_lr5e-7_batch256_avoidbias_filtered",
    "/fsx-onellm/stellalisy/health-q/ppo_models/llama-3.2-3b-instruct_lr5e-6_ep2/rm_lr9e-6_ppo_lr5e-7_batch256_clarity_filtered",
    "/fsx-onellm/stellalisy/health-q/ppo_models/llama-3.2-3b-instruct_lr5e-6_ep2/rm_lr9e-6_ppo_lr5e-7_batch256_focus_filtered",
    "/fsx-onellm/stellalisy/health-q/ppo_models/llama-3.2-3b-instruct_lr5e-6_ep2/rm_lr9e-6_ppo_lr5e-7_batch256_relevance_filtered"
]
output_directory = "/fsx-onellm/stellalisy/health-q/ppo_models/llama-3.2-3b-instruct_lr5e-6_ep2/rm_lr9e-6_ppo_lr5e-7_batch256_policyfusion_filtered"
average_model_weights(model_directories, output_directory, save_average=True)