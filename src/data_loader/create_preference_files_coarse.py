# huggingface-cli upload [dataset_repo_id] [local_path] [path_in_repo] --repo-type dataset

import json
import random
import jsonlines

more_key = "more_future" # "more"
less_key = "less"

system_prompt = "You are a medical doctor trying to interct with the patient to seek additional information about their case before making a clinical decision. Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, ask a follow up question. Base your answer on the current and standard practices referenced in medical guidelines.\nTask: You will be given some initial patient information and a patient inquiry, and you should ask a follow-up question to the patient. The question should be bite-sized, NOT ask for too much at once, and NOT repeat what has already been asked. Respond with the atomic question and NOTHING ELSE."

def build_counterfactual_pairs(data, features=["overall"]):
    """
    pair samples in data based on the counterfactual generations
    if the coversation does not have a conclusion, then we do not include it into the returned data
    return two lists of samples, one for accepted (has direct answer, which means no questions in convo) and one for rejected (has questions in convo)
    """
    samples = []
    count_per_feature = {f: 0 for f in features}
    for sample in data:
        original = sample["rewrite"]["content"]
        original = original.replace("Here are the rewritten questions:", "").strip()
        
        for feature in features:
            # append all feature pairs (more, less), (more, original), (less, original)
            system = [{"role": "system", "content": system_prompt}]
            context = [
                # {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{sample['title']}\n{sample['post']}"}
            ]
            for a, r in [(more_key, less_key), (more_key, "original"), ("original", less_key)]: 
                # append to accepted
                question_a = sample[feature][a]["content"] if a != "original" else original
                chosen = [{"role": "assistant", "content": question_a}]
                
                # append to rejected
                question_r = sample[feature][r]["content"] if r != "original" else original
                rejected = [{"role": "assistant", "content": question_r}]

                if feature == "overall": pair_id = f"{sample['id']}_coarse_{a}_{r}"
                else: pair_id = f"{sample['id']}_{feature}_{a}_{r}"

                samples.append({
                    "id": pair_id,
                    "system": system,
                    "context": context,
                    "chosen": chosen,
                    "rejected": rejected,
                })
                count_per_feature[feature] += 1
    print(f"Features: {features}\nNumber of pairs: {len(samples)}")
    print(count_per_feature)
    return samples


def split_save_data(data_file, output_file, write=False):  
    # split accepted rejected pairs to train, validation, test
    with open(data_file) as json_list:
        parsed_data = [json.loads(line) for line in json_list.readlines()]
    
    print(f"Number of samples: {len(parsed_data)}")

    pairs = build_counterfactual_pairs(parsed_data)
    print(f"Number of pairs: {len(pairs)}")
    
    if write:
        random.shuffle(pairs)
        with jsonlines.open(output_file, "w") as writer:
            writer.write_all(pairs)

data_dir = ""
input_filepath = f"{data_dir}counterfactual_questions/train_counterfactuals.jsonl"
output_filepath = f"preference/coarse/train.jsonl"
split_save_data(data_file=input_filepath, output_file=output_filepath, write=True)

input_filepath = f"{data_dir}/eval_counterfactuals.jsonl"
output_filepath = f"preference/coarse/validation.jsonl"
split_save_data(data_file=input_filepath, output_file=output_filepath, write=True)