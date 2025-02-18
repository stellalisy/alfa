# huggingface-cli upload [dataset_repo_id] [local_path] [path_in_repo] --repo-type dataset

import bz2
import io
import json
import lzma
import os
import copy
from os.path import isfile
from os.path import join as pjoin
from time import time

import datasets
from datasets.exceptions import DefunctDatasetError

import random
import jsonlines

more_key = "more_future" # "more"
less_key = "less"

system_prompt = "You are a medical doctor trying to interct with the patient to seek additional information about their case before making a clinical decision. Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, ask a follow up question. Base your answer on the current and standard practices referenced in medical guidelines.\nTask: You will be given some initial patient information and a patient inquiry, and you should ask a follow-up question to the patient. The question should be bite-sized, NOT ask for too much at once, and NOT repeat what has already been asked. Respond with the atomic question and NOTHING ELSE."

def build_synthetic_sft(data, good_ids, sft_data, features=["accuracy", "avoidbias", "relevance", "clarity", "focus", "answerability"]):
    """
    pair samples in data based on the counterfactual generations
    if the coversation does not have a conclusion, then we do not include it into the returned data
    return two lists of samples, one for accepted (has direct answer, which means no questions in convo) and one for rejected (has questions in convo)
    """
    data = {sample['id']: sample for sample in data}
    
    samples = []
    for qid, sample in sft_data.items():
        if qid not in data:
            # print(f"qid {qid} not in synthetic data")
            # breakpoint()
            continue
        
        sample['id'] = f"{qid}_original"
        samples.append(sample)

        couterfactual = data[qid]
        
        # original = sample["rewrite"]["content"]
        # original = original.replace("Here are the rewritten questions:", "").strip()
        
        # system = [{"role": "system", "content": system_prompt}]
        # context = [
        #         # {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": f"{sample['title']}\n{sample['post']}"}
        #     ]
        
        # samples.append({
        #     'id': f"{qid}_original",
        #     "system": system,
        #     "context": context,
        #     "question": [{"role": "assistant", "content": original}]
        # })

        for feature in features:
            # append all feature pairs (more, less), (more, original), (less, original)
            
            better_question = couterfactual[feature][more_key]["content"]
            synthetic_id = f"{qid}_{feature}"

            sample_temp = copy.deepcopy(sample)
            sample_temp['id'] = synthetic_id
            sample_temp['question'] = [{"role": "assistant", "content": better_question}]
            samples.append(sample_temp)

    return samples



def split_save_data(data_file, output_file, good_ids_file=None, sft_filepath=None, write=False):  
    # split accepted rejected pairs to train, validation, test
    with open(data_file) as json_list:
        parsed_data = [json.loads(line) for line in json_list.readlines()]
    print(f"Number of samples: {len(parsed_data)}")

    if good_ids_file:
        with open(good_ids_file) as f:
            good_ids = [line.strip().replace("rewrite", "original") for line in f.readlines()]
    else:
        good_ids = None
    
    if sft_filepath:
        with open(sft_filepath) as f:
            sft_data = [json.loads(line) for line in f.readlines()]
            sft_data = {sample['id']: sample for sample in sft_data}

    synthetic_data = build_synthetic_sft(parsed_data, good_ids, sft_data)
    print()
    
    if write:
        random.shuffle(synthetic_data)
        with jsonlines.open(output_file, "w") as writer:
            writer.write_all(synthetic_data)
        print(f"Saved to {output_file}")



data_dir = ""

good_ids_file = f"counterfactual_questions/verification/train_good_ids.jsonl"
input_filepath = f"counterfactual_questions/train_counterfactuals.jsonl"
output_filepath = f"sft/synthetic/train.jsonl"
sft_filepath = f"sft/train.jsonl"
split_save_data(data_file=input_filepath, output_file=output_filepath, good_ids_file=good_ids_file, sft_filepath=sft_filepath, write=True)

good_ids_file = f"counterfactual_questions/verification/eval_good_ids.jsonl"
input_filepath = f"counterfactual_questions/train_counterfactuals.jsonl"
output_filepath = f"sft/synthetic/validation.jsonl"
sft_filepath = f"sft/validation.jsonl"
split_save_data(data_file=input_filepath, output_file=output_filepath, good_ids_file=good_ids_file, sft_filepath=sft_filepath, write=True)

