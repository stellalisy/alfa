import os
import json
import logging
import argparse
from collections import defaultdict
from annotator import Annotator


# argparse
parser = argparse.ArgumentParser()
parser.add_argument("--annotators_config_filepath", type=str, default=True)  #/home/stellalisy/health-q/scripts/rank_eval/annotators/healthQ_llama405b_fn
parser.add_argument("--annotator_config_model_endpoint", type=str, default=None)
parser.add_argument("--annotator_config_self_consistency", type=int, default=None)

parser.add_argument("--reference_outputs_filepath", type=str, required=True)  #/home/stellalisy/health-q/data/rank_eval/test_reference.json
parser.add_argument("--reference_output_key", type=str, default="output_human")  
parser.add_argument("--model_outputs_filepath", type=str, required=True)    #/home/stellalisy/health-q/data/alpaca_eval/toy_outputs_text.json
parser.add_argument("--generator_output_key", type=str, default="output")

parser.add_argument("--annotation_filepath", type=str, required=True)    #/home/stellalisy/health-q/data/alpaca_eval/annotations/test_annotations.jsonl
parser.add_argument("--results_filepath", type=str, required=True)    #/home/stellalisy/health-q/scripts/rank_eval/results
parser.add_argument("--log_filepath", type=str, default=None)   #/home/stellalisy/health-q/scripts/rank_eval/log/test_log.log

parser.add_argument("--max_instances", type=int, default=1000000)
parser.add_argument("--best_of_n_idx", type=int, default=0)



args = parser.parse_args()

# output should be the same format as model_outputs, add a field "score" to store the score
os.makedirs(os.path.dirname(args.annotation_filepath), exist_ok=True)
os.makedirs(os.path.dirname(args.results_filepath), exist_ok=True)
os.makedirs(os.path.dirname(args.log_filepath), exist_ok=True)

# logging
logging.basicConfig(filename=args.log_filepath, level=logging.INFO,
                    format='[%(asctime)s - %(levelname)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
annotator_override_config = {}
for i, (key, value) in enumerate(vars(args).items()):
    if key.startswith("annotator_config_") and value is not None:
        annotator_override_config[key.replace("annotator_config_", "")] = value
    if i == len(vars(args).items())-1:
        logging.info(f"{key}: {value}\n========================================================================\n")
    else:
        logging.info(f"{key}: {value}")


# load reference outputs and model outputs
with open(args.reference_outputs_filepath, "r") as f:
    reference_outputs = json.load(f)
with open(args.model_outputs_filepath, "r") as f:
    if args.model_outputs_filepath.endswith(".json"): model_outputs = json.load(f)
    else: model_outputs = [json.loads(line.strip()) for line in f.readlines()]
model_outputs_by_id = defaultdict(dict)
for output in model_outputs:
    if output["best_of_n"] == args.best_of_n_idx:
        model_outputs_by_id[output["id"]][output["generator"]] = output


# load from checkpoint data
if os.path.exists(args.annotation_filepath):
    with open(args.annotation_filepath, "r") as f:
        annotation_data = [json.loads(line.strip()) for line in f.readlines()]
else: annotation_data = []
processed_data = defaultdict(dict)
"""
processed_data = {
    "t_dpon6tw-0-c2-q0" : {
        "generator_1": 0.5,
        "generator_2": 1.0,
    }
}
"""
for sample in annotation_data:
    thread_id = sample["id"]
    generator_name = sample["generator"]
    score = sample["score"]
    processed_data[thread_id][generator_name] = score


# initialize annotator
annotator = Annotator(args.annotators_config_filepath, override_config=annotator_override_config)


generator_scores = defaultdict(list)
for ref_i, reference in enumerate(reference_outputs):
    outputs_dict = model_outputs_by_id[reference["id"]] # outputs from each model {"generator_name": output_text}

    logging.info(f"[Instruction]\n{reference['instruction']}")
    logging.info(f"[Reference] {reference[args.reference_output_key]}")

    for generator_name, generator_output in outputs_dict.items():
        
        processed_score = processed_data[reference["id"]][generator_name] if generator_name in processed_data[reference["id"]] else None
        
        logging.info(f"[Model Res] {generator_output['output']}")
            
        if processed_score:
            score = processed_score
            source = "Read from proccessed data"
        else:
            annotation = annotator.annotate_pair(reference["instruction"], reference[args.reference_output_key], generator_output[args.generator_output_key])
            generator_output["reference"] = reference[args.reference_output_key]
            for key, value in annotation.items(): generator_output[key] = value
            with open(args.annotation_filepath, "a") as f:
                f.write(json.dumps(generator_output) + "\n")
            score = annotation["score"]
            source = f"Labeled by {annotator.annotator_name}"

        logging.info(f"Generator: {generator_name}, Score: {score} <== {source}\n")
        if score: generator_scores[generator_name].append(score)
    
    logging.info(f"Proccessed {ref_i+1} instances (reference id: {reference['id']})")
    for i, (generator, scores) in enumerate(generator_scores.items()):
        if i == len(generator_scores)-1: logging.info(f"Generator: {generator}, Average score: {sum(scores) / len(scores)}\n========================================================================\n")
        else: logging.info(f"Generator: {generator}, Average score: {sum(scores) / len(scores)}")
    
    if ref_i >= args.max_instances:
        break

results = {}
for generator, scores in generator_scores.items():
    logging.info(f"Generator: {generator}, Average score: {sum(scores) / len(scores)}")
    results[generator] = {"average_score": sum(scores) / len(scores), "scores": scores}
    
with open(args.results_filepath, "w") as f:
    json.dump(results, f, indent=2)