import os
import json
from collections import defaultdict
from question_generation_prompt_open import template, system_prompt
from openai import OpenAI
from keys import API_KEY

client = OpenAI(api_key=API_KEY["healthQ"])

input_filepath = "/home/stellalisy/health-q/data/ids/test_q.jsonl"

with open(input_filepath, "r") as f:
    data = f.readlines()
data = [json.loads(d) for d in data]

threads = defaultdict(list)
useless_columns = ["author_post", "id_post", "post_author_feedback", "pos_post_author_feedback", "turn_count", "total_score", "verified_author_flairs", "unique_author_counts", "year", "created_utc", "has_conclusion", "refer_to_hospital", "comment_id", "comment_author", "comment_is_question", "comment_is_response", "comment_respond_to", "comment_requested_info", "comment_question_type", "comment_is_personal_experience", "comment_is_medical_knowledge", "comment_is_answered", "comment_is_relevant", "comment_score", "comment_parent_id", "comment_turn_level", "comment_author_flair_text", "comment_created_utc", "question_type", "binary_question", "has_final_diagnosis"]

for sample in data:
    for column in useless_columns:
        if column in sample: sample.pop(column, None)
    threads[sample["thread_id"]].append(sample)

out_filename = "/home/stellalisy/health-q/data/mediq_eval/test_qs.jsonl"
if os.path.exists(out_filename):
    with open(out_filename, "r") as f:
        existing_data = f.readlines()
    existing_data = [json.loads(d) for d in existing_data]
    existing_threads = [d["thread_id"] for d in existing_data]
else:
    existing_threads = []

usage = defaultdict(int)
for thread_id, samples in threads.items():

    if thread_id in existing_threads:
        print(f"Thread {thread_id} already processed")
        continue

    post = samples[0]["post"]
    title = samples[0]["title"]
    
    questions = [sample["question"] if "question" in sample and sample["question"] is not None else sample["comment_body"] for sample in samples]
    responses = [sample["patient_answer"] for sample in samples]
    convo = [{"Doctor Question": q, "Patient Response": r} for q, r in zip(questions, responses)]
    
    hypotheses = set()
    for sample in samples:
        if "hypotheses" in sample:
            hypotheses.update(sample["hypotheses"])
    hypotheses = list(hypotheses)
    
    final_diagnosis = samples[0]["final_diagnosis"]
    conclusion = samples[0]["conclusion"]
    
    input = {
        "Title": title,
        "Initial information": post,
        "Additional information": convo,
        "Hypotheses": hypotheses,
        "Final Diagnosis": final_diagnosis,
        "Conclusion": conclusion
    }

    prompt = "Patient Information:\n\n"+json.dumps(input, indent=4)+template

    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="o1",
        messages=message,
        response_format={
            "type": "json_object"
        }
    )

    response_text = response.choices[0].message.content
    try:
        mcqs = json.loads(response_text)
    except:
        mcqs = response_text
        print("Error in parsing the response")
    
    try:
        parsed_questions = []
        for i in range(1,4):
            question = {"question": mcqs[f"question{i}"], 
                        "options": {j: mcqs[f"question{i}_option{j}"] for j in "ABCD"}, "correct_answer": mcqs[f"question{i}_correct_answer"]}
            parsed_questions.append(question)
    except:
        print("Error in parsing the questions")
        parsed_questions = mcqs
    
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens

    output_obj = {
        "thread_id": thread_id,
        "input": input,
        "output": mcqs,
        "parsed_output": parsed_questions,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens
    }
    usage["input_tokens"] += response.usage.prompt_tokens,
    usage["output_tokens"] += response.usage.completion_tokens,
    usage["reasoning_tokens"] += response.usage.completion_tokens_details.reasoning_tokens

    # save the generated questions -- save the entire response variable
    with open(out_filename, "a") as f:
        f.write(json.dumps(output_obj))
        f.write("\n")

    print(f"Thread {thread_id} processed, usage: {usage}")