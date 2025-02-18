import os
import json
import argparse
from collections import defaultdict
from openai import OpenAI
from keys import API_KEY


def convert_to_mediq_o1():
    with open(input_filename, "r") as f:
        data = f.readlines()
    data = [json.loads(d) for d in data]

    if os.path.exists(output_filename):
        with open(output_filename, "r") as f:
            processed_data = f.readlines()
        processed_ids = [json.loads(d)["id"] for d in processed_data]
    else:
        processed_ids = []

    usage = defaultdict(int)
    for d in data:
        pid = d["thread_id"]
        if pid in processed_ids:
            continue
        question = d["parsed_output"]["question"]
        patient_info = d["input"]["Title"] + "\n\n" + d["input"]["Initial information"] + "\n\n" + "\n\n".join([f"Doctor Question: {turn['Doctor Question']}\nPatient Response: {turn['Patient Response']}" for turn in d["input"]["Additional information"]])
        options = {
            "A": d["parsed_output"]["optionA"],
            "B": d["parsed_output"]["optionB"],
            "C": d["parsed_output"]["optionC"],
            "D": d["parsed_output"]["optionD"]
        }
        answer_idx = d["parsed_output"]["correct_answer"]
        answer = options[answer_idx]
        atomic_facts, sample_usage = get_atomic_facts(patient_info)
        output = {
            "id": pid,
            "question": question,
            "options": options,
            "answer": answer,
            "answer_idx": answer_idx,
            "atomic_facts": atomic_facts,
            "usage": sample_usage
        }
        for k, v in sample_usage.items(): usage[k] += v
        with open(output_filename, "a+") as f:
            f.write(json.dumps(output) + "\n")
        print(f"Thread {pid} processed, usage: {sample_usage}")
    print(f"Total usage: {usage}")

def get_atomic_facts(patient_info):
    prompt = f"Break the following patient record into a list of independent atomic facts about the patient, with one piece of information in each statement. Each fact should only include the smallest unit of information, but should be self-contained.\n\n****START OF PATIENT RECORD***\n\"{patient_info}\"\n****END OF PATIENT RECORD***\n\nResponse with the list of atomic facts and nothing else. Use '-' as the bullet of each fact, one line per fact. No sub-list allowed."
    messages = [
        {
            "role": "system",
            "content": "You are a helpful medical assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    response = client.chat.completions.create(model=args.model_name, messages=messages)
    atomic_facts = [s.strip() for s in response.choices[0].message.content.strip().split("\n")]

    
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "reasoning_tokens": response.usage.completion_tokens_details.reasoning_tokens
    }

    return atomic_facts, usage


def posthoc_edit_add_context_and_initial_info():
    # assume that output file is already generated using the convert_to_mediq_o1() function
    with open(input_filename, "r") as f:
        input_data = f.readlines()
    input_data = [json.loads(d) for d in input_data]
    input_data = {d["thread_id"]: d for d in input_data}

    with open(output_filename, "r") as f:
        output_data = f.readlines()
    output_data = [json.loads(d) for d in output_data]
    output_data = {d["id"]: d for d in output_data}

    usage = defaultdict(int)
    for pid, d in output_data.items():
        input_sample = input_data[pid]
        original_post = input_sample["input"]["Title"] + "\n\n" + input_sample["input"]["Initial information"]
        patient_info = original_post + "\n\n" + "\n\n".join([f"Doctor Question: {turn['Doctor Question']}\nPatient Response: {turn['Patient Response']}" for turn in input_sample["input"]["Additional information"]])
        initial_info, sample_usage = extract_initial_info(patient_info)
        d["context"] = patient_info
        d["initial_info"] = initial_info
        for k, v in sample_usage.items(): d['usage'][k] += v
        with open(output_filename_v2, "a+") as f:
            f.write(json.dumps(d) + "\n")
        print(f"Thread {pid} processed, usage: {sample_usage}")
    print(f"Total usage: {usage}")
        

def extract_initial_info(patient_info):
    task_instruction = "Below is a sample reddit post from r/Askdocs where a user is trying to seek some medical advice:\n\n{patient_info}\n\nYour task is to summarize this into one sentence that represents a typical chief complaint of a patient - what would this patient say to a clinician when they first walk into the clinic? Respond with the chief complaint only and nothing else."
    messages = [
        {
            "role": "system",
            "content": "You are a helpful medical assistant."
        },
        {
            "role": "user",
            "content": task_instruction
        }
    ]
    response = client.chat.completions.create(model=args.model_name, messages=messages)
    initial_info = response.choices[0].message.content.strip()

    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "reasoning_tokens": response.usage.completion_tokens_details.reasoning_tokens
    }

    return initial_info, usage



def add_original_post_to_sample():
    # assume that output file is already generated using the convert_to_mediq_o1() and posthoc_edit_add_context_and_initial_info() functions
    with open(input_filename, "r") as f:
        input_data = f.readlines()
    input_data = [json.loads(d) for d in input_data]
    input_data = {d["thread_id"]: d for d in input_data}

    with open(output_filename_v2, "r") as f:
        output_data = f.readlines()
    output_data = [json.loads(d) for d in output_data]
    output_data = {d["id"]: d for d in output_data}

    for pid, d in output_data.items():
        input_sample = input_data[pid]
        original_post = input_sample["input"]["Title"] + "\n\n" + input_sample["input"]["Initial information"]
        d["original_post"] = original_post
        with open(output_filename_v3, "a+") as f:
                f.write(json.dumps(d) + "\n")
    print("Done")



# the same as the above 3 steps but all in one function
def main():
    with open(input_filename, "r") as f:
        data = f.readlines()
    data = [json.loads(d) for d in data]

    if os.path.exists(output_filename_v3):
        with open(output_filename_v3, "r") as f:
            processed_data = f.readlines()
        processed_ids = [json.loads(d)["id"] for d in processed_data]
    else:
        processed_ids = []

    usage = defaultdict(int)
    for d in data:
        pid = d["thread_id"]
        if pid in processed_ids:
            continue
        
        original_post = d["input"]["Title"] + "\n\n" + d["input"]["Initial information"]
        patient_info = original_post + "\n\n" + "\n\n".join([f"Doctor Question: {turn['Doctor Question']}\nPatient Response: {turn['Patient Response']}" for turn in d["input"]["Additional information"]])
        
        initial_info, sample_usage = extract_initial_info(patient_info)
        atomic_facts, sample_usage = get_atomic_facts(patient_info)
        
        question = d["parsed_output"]["question"]
        options = {
            "A": d["parsed_output"]["optionA"],
            "B": d["parsed_output"]["optionB"],
            "C": d["parsed_output"]["optionC"],
            "D": d["parsed_output"]["optionD"]
        }
        answer_idx = d["parsed_output"]["correct_answer"]
        answer = options[answer_idx]
        output = {
            "id": pid,
            "question": question,
            "options": options,
            "answer": answer,
            "answer_idx": answer_idx,
            "initial_info": initial_info,
            "original_post": original_post,
            "context": patient_info,
            "atomic_facts": atomic_facts,
            "usage": sample_usage
        }
        for k, v in sample_usage.items(): usage[k] += v
        with open(output_filename_v3, "a+") as f:
            f.write(json.dumps(output) + "\n")
        print(f"Thread {pid} processed, usage: {sample_usage}")
    print(f"Total usage: {usage}")


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="o1")
args = parser.parse_args()

client = OpenAI(api_key=API_KEY["healthQ"])

input_filename = "/gscratch/argon/stelli/health-q/data/mediq_eval/test_qs_post.jsonl"
output_filename = "/gscratch/argon/stelli/health-q/data/mediq_eval/test_qs_post_mediq.jsonl"
output_filename_v2 = "/gscratch/argon/stelli/health-q/data/mediq_eval/test_qs_post_mediq_v2.jsonl"
output_filename_v3 = "/gscratch/argon/stelli/health-q/data/mediq_eval/test_qs_post_mediq_v3.jsonl"

if __name__ == "__main__":
    main()