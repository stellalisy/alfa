import os
import subprocess
import yaml
import argparse

ROOT_DIR="."
DEFAULT_REFERENCE="test_reference.json"
DEFAULT_ANNOTATOR="healthQ_llama405b_cot_clf"

parser = argparse.ArgumentParser(description="Launch rank eval pipeline with specified configuration.")
parser.add_argument("--run_config_filepath", type=str, required=True, help="Path to the YAML configuration file")
parser.add_argument("--root_dir", type=str, default=ROOT_DIR, help="Root directory for the rank eval pipeline")
parser.add_argument("--script_dir", type=str, default=None, help="Root directory for the rank eval pipeline")
parser.add_argument("--data_dir", type=str, default=None, help="Data directory for the rank eval pipeline")
parser.add_argument("--script_filepath", type=str, default="rank_eval.py", help="Path to the rank eval script")
parser.add_argument("--default_annotator", type=str, default=DEFAULT_ANNOTATOR, help="Default annotator to use if not specified in the config")
parser.add_argument("--annotator_config_self_consistency", type=int, default=None)

parser.add_argument("--annotator_key", type=str, default="")
parser.add_argument("--reference_output_key", type=str, default="output")
parser.add_argument("--generator_output_key", type=str, default="output")
parser.add_argument("--best_of_n_idx", type=int, default=0)
args = parser.parse_args()

if not args.script_dir: args.script_dir = os.path.join(args.root_dir, "scripts", "rank_eval")
if not args.data_dir: args.data_dir = os.path.join(args.root_dir, "data", "rank_eval")
delattr(args, "root_dir")

with open(args.run_config_filepath, "r") as f:
    config = yaml.safe_load(f)
delattr(args, "run_config_filepath")


reference_outputs_filepath = os.path.join(args.data_dir, config.pop("reference_outputs_filepath", DEFAULT_REFERENCE))
annotators_config_filepath = os.path.join(args.script_dir, "annotators", config.pop("annotators_config_filepath", args.default_annotator))
delattr(args, "default_annotator")
if not annotators_config_filepath.endswith(".yaml"): annotators_config_filepath = os.path.join(annotators_config_filepath, "configs.yaml")

exp_group = config.pop("exp_group", "test_group")
exp_key = config.pop("exp_key", "test_key")
if args.annotator_key != "": args.annotator_key = '_' + args.annotator_key
model_outputs_filepath = os.path.join(args.data_dir, "model_generations", exp_group, f"{exp_key}.jsonl")
annotation_filepath = os.path.join(args.data_dir, "annotations", exp_group, f"{exp_key}{args.annotator_key}_annotations.jsonl")
results_filepath = os.path.join(args.script_dir, "results", f"{exp_group}_{exp_key}{args.annotator_key}_results.json")
log_filepath = os.path.join(args.script_dir, "log", exp_group, f"{exp_key}{args.annotator_key}.log")
delattr(args, "data_dir")
delattr(args, "annotator_key")

run_commant = "python"
script_path = os.path.join(args.script_dir, args.script_filepath)
delattr(args, "script_dir")
delattr(args, "script_filepath")
command = [run_commant, script_path]
command.extend(["--annotators_config_filepath", annotators_config_filepath])
command.extend(["--reference_outputs_filepath", reference_outputs_filepath])
command.extend(["--model_outputs_filepath", model_outputs_filepath])
command.extend(["--annotation_filepath", annotation_filepath])
command.extend(["--results_filepath", results_filepath])
command.extend(["--log_filepath", log_filepath])
for key, value in vars(args).items():
    command.extend([f"--{key}", str(value)])

for key, value in config.items():
    command.append(f"--{key}")
    command.append(str(value))
    
subprocess.run(command)