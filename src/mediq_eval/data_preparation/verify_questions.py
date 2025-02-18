import os
import json
from collections import defaultdict
import argparse
from question_verification_prompt import template, system_prompt
from openai import OpenAI
from keys import API_KEY

# set up arg parser
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--sc_index", type=str, required=True)
args = parser.parse_args()

client = OpenAI(api_key=API_KEY["healthQ"])

input_filepath = "/gscratch/argon/stelli/health-q/data/mediq_eval/test_qs_post.jsonl"

with open(input_filepath, "r") as f:
    data = f.readlines()
data = [json.loads(d) for d in data]
questions = {d["thread_id"]:d for d in data}

out_filename = f"/gscratch/argon/stelli/health-q/data/mediq_eval/test_qs_post_verify_{args.model_name}_{args.sc_index}.jsonl"

if os.path.exists(out_filename):
    with open(out_filename, "r") as f:
        existing_data = f.readlines()
    existing_data = [json.loads(d) for d in existing_data]
    accuracy, accuracy_correct = existing_data[-1]["accuracy"], existing_data[-1]["accuracy_on_correct"]
    good_ls = [1] * int(accuracy * len(existing_data)) + [0] * (len(existing_data) - int(accuracy * len(existing_data)))
    correct_good_ls = [1] * int(accuracy_correct * len(existing_data)) + [0] * (len(existing_data) - int(accuracy_correct * len(existing_data)))
    processed_ids = [d["thread_id"] for d in existing_data]
else:
    processed_ids = []
    good_ls, correct_good_ls = [], []


def main():
    usage = defaultdict(int)
    for thread_id, sample in questions.items():

        if thread_id in processed_ids:
            print(f"Thread {thread_id} already processed")
            continue

        patient = sample["input"]
        question = sample["parsed_output"]["question"]
        options = [sample["parsed_output"][f"option{i}"] for i in "ABCD"]
        correct_answer = sample["parsed_output"]["correct_answer"].replace("Option ", "").replace("option ", "")
        correct_answer_idx = "ABCD".index(correct_answer)

        output = verify_mcq(patient, question, options, correct_answer_idx, usage)
        output["thread_id"] = thread_id

        good_ls.append(output["good"])
        correct_good_ls.append(output["verifier"][correct_answer_idx])
        output["accuracy"] = sum(good_ls) / len(good_ls)
        output["accuracy_on_correct"] = sum(correct_good_ls) / len(correct_good_ls)

        with open(out_filename, "a+") as f:
            f.write(json.dumps(output) + "\n")

        print(f"Thread {thread_id} processed, accuracy: {output['accuracy']}, accuracy_on_correct: {output['accuracy_on_correct']}, usage: {usage}")


def verify_mcq(patient, question, options, correct_answer_idx, usage):
    verifier = []
    sample_usage = defaultdict(int)
    for i in range(4):
        verify_output = verify_option(patient, question, options[i], i == correct_answer_idx, sample_usage)
        verifier.append(verify_output["decision"])
    for k, v in sample_usage.items():
        usage[k] += v
    out_dict = {
        "verifier": verifier,
        "good": all(verifier),
        "usage": sample_usage
    }
    return out_dict


def verify_option(patient, question, option, correct_option, usage):
    prompt = template.format(patient=json.dumps(patient, indent=4), 
                             question=question, 
                             option=option)

    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model=args.model_name,
        messages=message,
    )

    response_text = response.choices[0].message.content
    decision = "[yes]" in response_text if correct_option else "[no]" in response_text
    if ("[yes]" in response_text and "[no]" in response_text) or not ("[yes]" in response_text or "[no]" in response_text):
        decision = None
    
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens

    usage["input_tokens"] += input_tokens
    usage["output_tokens"] += output_tokens
    usage["reasoning_tokens"] += reasoning_tokens

    output_obj = {
        "output": response_text,
        "decision": decision,
    }

    return output_obj

if __name__ == "__main__":
    main()