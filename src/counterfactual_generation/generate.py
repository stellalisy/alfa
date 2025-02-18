import os
import json
import logging
import sys
import time
from collections import defaultdict
import argparse
import warnings
import copy
import time
warnings.filterwarnings('ignore')

import prompts as rewrite_prompts

import sys
sys.path.append('/data/home/stellalisy/social_alignment/src')

from llama_405b_client import Llama3_Client


parser = argparse.ArgumentParser(prog='ProgramName', description='What the program does', epilog='Text at the bottom of help')
parser.add_argument('--dimension', type=str, default='accuracy,avoidbias,relevance,clarity,focus,answerability')  # ['accuracy', 'avoidbias', 'relevance', 'clarity', 'focus', 'answerability']
parser.add_argument('--input_data', type=str, default="/fsx-sage/stellalisy/social_alignment/askdocs/data/inputs/askdocs_threads_all.jsonl")
parser.add_argument('--target_qids_filepath', type=str, default=None)  #/fsx-sage/stellalisy/social_alignment/askdocs/data/ids/eval_qids.txt
parser.add_argument('--prompts_dir', type=str, default="/home/stellalisy/health-q/data/prompts")
parser.add_argument('--output_filepath', type=str, default=None)
parser.add_argument('--log_filepath', type=str, default="/home/stellalisy/social_alignment/src/counterfactual_generation/log/counterfactual_generation.log")
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.1-405B-Instruct-FP8")
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--max_tokens', type=int, default=512)
parser.add_argument('--model_endpoint', type=str, default="http://10.200.58.137:7023")
parser.add_argument('--timeout', type=int, default=10)
parser.add_argument('--rewrite_fewshot', action='store_true')
args = parser.parse_args()

logging.basicConfig(filename=args.log_filepath, level=logging.INFO, 
                    format='[%(asctime)s - %(levelname)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
for arg in vars(args):
    logging.info(f"{arg}: {getattr(args, arg)}")

client = Llama3_Client(model_endpoint=args.model_endpoint, model_name=args.model_name)

def transform_comments(dimensions=args.dimension):
    num_questions_each_turn_stats = defaultdict(int)
    if dimensions == "none":
        dimensions = []
    elif isinstance(dimensions, str):
        if ',' in dimensions:
            dimensions = dimensions.split(',')
        else:
            dimensions = [dimensions]

    print(f"Generating for dimension: {dimensions}")
    logging.info(f"Generating for dimension: {dimensions}")

    dimension_prompts = read_dimension_prompts(args.prompts_dir, dimensions)
    # load original comments data 
    with open(args.input_data, "r") as f:
        threads = [json.loads(line) for line in f.readlines()]

    if os.path.exists(args.output_filepath):
        with open(args.output_filepath, "r") as f:
            prompt_data = f.readlines()
        data_processed = [json.loads(line) for line in prompt_data]
        data_processed = {sample['id']: sample for sample in data_processed}

        backup_filepath = args.output_filepath.replace('.jsonl', f'.{time.strftime("%Y%m%d-%H%M%S")}.jsonl')
        os.rename(args.output_filepath, backup_filepath)
        sys.stdout.write("backed up existing file to " + backup_filepath + "\n")
        logging.info("backed up existing file to " + backup_filepath)
    else: 
        data_processed = {}

    if args.target_qids_filepath is not None:
        with open(args.target_qids_filepath, "r") as f:
            target_qids = [line.strip() for line in f.readlines()]
    else:
        target_qids = None
    
    # thread_ids = []
    usage = {"prompt_tokens": 0, "total_tokens": 0, "completion_tokens": 0}

    for i in range(len(threads)):
        sample = copy.deepcopy(threads[i])
        thread_id = sample['thread_id']
        # thread_id = f"{sample['thread_id']}-{thread_ids.count(sample['thread_id'])}"
        # thread_ids.append(sample['thread_id'])
            
        post = sample['post']
        title = sample['title']

        final_diagnosis = sample['final_diagnosis']
        final_diagnosis = "None" if final_diagnosis is None else final_diagnosis
        final_conclusion = sample['conclusion']
        final_conclusion = "None" if final_conclusion is None else final_conclusion

        for turn_idx, turn in enumerate(sample['convo']):
            if "questions" not in sample['convo'][turn_idx]:
                continue

            questions = [q['question'] for q in sample['convo'][turn_idx]['questions']]
            num_questions_each_turn_stats[turn_idx] += 1
            
            sys.stdout.write("======== Processing thread " + thread_id + ", " + str(turn_idx) + "th turn: # questions = " + str(len(questions)) + " ========\n")
            logging.info("======== Processing thread " + thread_id + ", " + str(turn_idx) + "th turn: # questions = " + str(len(questions)) + " ========")

            for q_idx, question in enumerate(questions):
                question_id = thread_id + f"-c{turn_idx}-q{q_idx}"
                if args.target_qids_filepath is not None and question_id not in target_qids: continue

                sys.stdout.write("------------ Question #" + str(q_idx) + ": " + question_id + " ------------\n")
                sys.stdout.write("original: " + question + "\n")
                logging.info("------------ Question #" + str(q_idx) + ": " + question_id + " ------------")
                logging.info("original: " + question)
                out_dict, usage = process_question(question=question, 
                                            question_id=question_id,
                                            title=title, post=post, 
                                            data_processed=data_processed, 
                                            dimensions=dimensions,
                                            dimension_prompts=dimension_prompts,
                                            usage=usage, 
                                            final_diagnosis=final_diagnosis,
                                            conclusion=final_conclusion, turn=turn)
                if 'title' not in out_dict: out_dict['title'] = title
                if 'post' not in out_dict: out_dict['post'] = post
                os.makedirs(os.path.dirname(os.path.realpath(args.output_filepath)), exist_ok=True)
                with open(args.output_filepath, "a") as f:
                    json.dump(out_dict, f)
                    f.write("\n")


def process_question(question, question_id, title, post, data_processed, dimensions, dimension_prompts, usage, final_diagnosis=None, conclusion=None, turn=None):
    out_dict = defaultdict(dict)
    
    if question_id in data_processed:
        sample = data_processed[question_id]
        if "original" not in sample: sample["original"] = question
        if all([dim in sample and 'less' in sample[dim] and 'more' in sample[dim] for dim in dimensions] + ["rewrite" in sample]):
            return sample, usage
        out_dict = defaultdict(dict, sample)
    else:
        out_dict['id'] = question_id
    out_dict['original'] = question

    # rewrite the question into a llama3-405b rewrite first
    if "rewrite" not in out_dict:
        message = []
        message.append({"role": "system", "content": rewrite_prompts.system})
        if args.rewrite_fewshot:
            message.extend(rewrite_prompts.singl_q_few_shot_example)
        message.append({"role": "user", "content": rewrite_prompts.single_q_user.format(title=title, post=post, comment_body=turn['body'], parsed_question=question)})
        temperature = 0.6
        response_obj = client.chat_completion(message, 
                                            model_name=args.model_name, 
                                            max_tokens=args.max_tokens, 
                                            temperature=temperature)
        assert response_obj["status_ok"]
        response_obj.pop("status_ok")
        rewritten_question = response_obj["content"]
        sys.stdout.write("rewritten: " + rewritten_question + "\n")
        logging.info("rewritten: " + rewritten_question)
        for k in response_obj["usage"]: 
            if k in usage:
                usage[k] += response_obj["usage"][k] 
        out_dict['rewrite'] = response_obj
    else:
        rewritten_question = out_dict['rewrite']['content']
                
    for dim in dimensions:
        for rel in ['less', 'more', 'more_future']:
            if dim in out_dict and rel in out_dict[dim]: continue
            instruction = dimension_prompts[dim][rel]
            system_prompt = instruction.split("\n\n\n")[0]
            prompt_template = instruction.split("\n\n\n")[1]
            if rel == "more_future":
                prompt = prompt_template.format(title=title, post=post, question=rewritten_question, final_diagnosis=final_diagnosis, conclusion=conclusion)
            else:
                prompt = prompt_template.format(title=title, post=post, question=rewritten_question)
                    
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}]
                    
            response_obj = client.chat_completion(messages, 
                                            model_name=args.model_name, 
                                            max_tokens=args.max_tokens, 
                                            temperature=args.temperature)

            assert response_obj["status_ok"]
            response_obj.pop("status_ok")

            response_text = response_obj["content"]
            for k in response_obj["usage"]: 
                if k in usage:
                    usage[k] += response_obj["usage"][k]
            
            sys.stdout.write(rel + " " + dim + ": " + response_text + "\n")
            logging.info(rel + " " + dim + ": " + response_text)
            out_dict[dim][rel] = response_obj
    return out_dict, usage

def read_dimension_prompts(prompts_dir, dimensions):
    dimension_prompts = defaultdict(dict)
    for dim in dimensions:
        for rel in ['less', 'more', 'more_future']:
            with open(os.path.join(prompts_dir, f"{dim}_{rel}.txt"), "r") as f:
                dimension_prompts[dim][rel] = f.read()
    return dimension_prompts
# test_model_worker(url=args.model_endpoint, model_name=args.model_name)

transform_comments()
