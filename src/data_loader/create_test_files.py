# huggingface-cli upload [dataset_repo_id] [local_path] [path_in_repo] --repo-type dataset

import json
import os
from os.path import join as pjoin
import random
import jsonlines

system_prompt = "You are a medical doctor trying to interct with the patient to seek additional information about their case before making a clinical decision. Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, ask a follow up question. Base your answer on the current and standard practices referenced in medical guidelines.\nTask: You will be given some initial patient information and a patient inquiry, and you should ask a follow-up question to the patient. The question should be bite-sized, NOT ask for too much at once, and NOT repeat what has already been asked. Respond with the atomic question and NOTHING ELSE."

def build_sft_samples(data):
    """
    prompt - completion sft data according to:
    https://huggingface.co/docs/trl/en/sft_trainer
    """
    samples = []
    # for sample in data:
    #     text = f'### User: {sample["title"]}\n{sample["post"]}\n### Assistant:'
    #     for ctxt in sample["prev_contexts"]:
    #         if "question" not in ctxt or "patient_answer" not in ctxt:
    #             continue
    #         text += f' {ctxt["question"]}\n### User: {ctxt["patient_answer"]}\n### Assistant:'
    #     text += f' {sample["question"]}'
    #     samples.append({'id': f'{sample["qid"]}', 'text': text})
    
    # format to messages for chat template
    conv_lens = []
    for sample in data:
        system = {"role": "system", "content": system_prompt},
        context = [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{sample['title']}\n{sample['post']}"}
            ]
        for ctxt in sample["prev_contexts"]:
            if "question" not in ctxt or "patient_answer" not in ctxt:
                continue
            context.append({"role": "assistant", "content": f'{ctxt["question"]}'})
            context.append({"role": "user", "content": f'{ctxt["patient_answer"]}'})
        question = [{"role": "assistant", "content": f'{sample["question"]}'}]
        messages = context + question
        samples.append({'id': f'{sample["qid"]}', 'system': system, 'messages': messages, 'context': context, 'question': question})
        conv_lens.append(len(messages))
    print(f"Number of samples: {len(samples)}, average conversation length: {sum(conv_lens)/len(conv_lens)}")
    return samples

def split_save_data(data_file="sft/askdocs_parsed_attributes_qa_flattened.jsonl", num_samples=None):  
    # split accepted rejected pairs to train, validation, test
    with open(data_file) as json_list:
        parsed_data = [json.loads(line) for line in json_list.readlines()]
    
    data_dir = os.path.dirname(data_file)
    print(f"Number of samples before excluding: {len(parsed_data)}")
    target_post_ids = []
    with open(pjoin(target_dir, target_file)) as f:
        target_post_ids.extend([s.strip() for s in f.readlines()])
    target_post_ids = set(target_post_ids)
    print("Number of post ids to process: ", len(target_post_ids))

    parsed_data = [sample for sample in parsed_data if sample["id_post"].strip() in target_post_ids]
    
    random.shuffle(parsed_data)
    if num_samples != None and num_samples < len(parsed_data):
        parsed_data = random.sample(parsed_data, num_samples)
    data_samples = build_sft_samples(parsed_data)

    # save the split data to a file
    with jsonlines.open(target_output_filename, "w") as writer:
        writer.write_all(data_samples)


target_file = "test_post_ids.txt"
target_dir = "split_ids"
target_output_filename = "rank_eval/test.jsonl"

split_save_data()