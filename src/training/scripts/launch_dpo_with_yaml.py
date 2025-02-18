import os
import subprocess
import torch
import yaml
import sys
import time
import argparse

# ROOT_DIR = '/gscratch/argon/stelli/health-q/src'
ROOT_DIR = '/home/stellalisy/health-q/scripts'
OPENRLHF_DIR = f'{ROOT_DIR}/training/OpenRLHF'
OPENINSTRUCT_DIR = f'{ROOT_DIR}/training/open-instruct'
MODEL_OUTPUT_DIR = os.path.join(OPENRLHF_DIR, "outputs")
RESULTS_DIR = "/mmfs1/gscratch/stf/stelli/health-q/results"

TRAIN_COMMAND = "openrlhf.cli.train_dpo"
LORA_MERGE_PATH = "open_instruct/merge_lora.py"
EVAL_SCRIPT_PATH = f"{ROOT_DIR}/training/scripts/launch_single_model_evals_no_revision.py"

def parse_args():
    parser = argparse.ArgumentParser(description="Launch training pipeline with specified configuration.")
    parser.add_argument('--master_port', type=int, default=29500, help='Master port for distributed training')
    parser.add_argument('--deepspeed_stage', type=int, choices=[0, 1, 2, 3], default=None, help='DeepSpeed stage')
    parser.add_argument('--num_gpus', type=int, default=8, help='Number of GPUs to use')
    parser.add_argument('--train_yaml_path', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--train_overrides', type=str, default=None, help='Overrides for the training config')
    parser.add_argument('--eval_list', type=str, nargs='+', default=["gsm", "math", "ifeval", "bbh", "truthfulqa", "alpaca"], help='List of evaluations to run')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluations')
    parser.add_argument('--skip_eval_if_exists', action='store_true', help='Skip evaluations on datasets where the results already exist')

    args = parser.parse_args()
    return args

def launch_training_pipeline(args, config):
    model_path = os.path.join(OPENRLHF_DIR, config['save_path'])

    if args.eval_only:
        return model_path
 
    curr_dir = os.getcwd()
    os.chdir(OPENRLHF_DIR)


    print(f"Training model using {args.num_gpus} GPUs")
    print(f"Training config: {args.train_yaml_path}")
    print(f"DeepSpeed stage: {args.deepspeed_stage}")

    cuda_visible_devices = None
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        print(f"Using CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
        del os.environ['CUDA_VISIBLE_DEVICES']

    if args.deepspeed_stage is not None:
        config["zero_stage"] = args.deepspeed_stage

    if args.train_overrides is not None:
        for override in args.train_overrides.split(","):
            if '=' in override:
                key, value = override.split('=')
                if value.lower() == "true":
                    config[key] = True
                elif value.lower() == "false" and key in config:
                    del config[key]
                else:
                    config[key] = value
            else:
                config[override] = True
    training_args = []
    for k, v in config.items():
        # Skip the keys that have false values
        if v is False:
            continue
        training_args.append(f"--{k}")
        # if its a --store_true argument, we don't need to add the value
        if v is True or v is None:
            continue
        # otherwise, add the value
        training_args.append(str(v))

    training_command = ['deepspeed', '--master_port', str(args.master_port)]
    if cuda_visible_devices is not None:
        training_command.extend(['--include', 'localhost:' + cuda_visible_devices])
    
    training_command += ['--module', TRAIN_COMMAND]
    training_command += training_args

    print(f"Running command: {' '.join(training_command)}")
    subprocess.run(training_command, env=os.environ, stdout=sys.stdout, stderr=subprocess.STDOUT)
    
    if cuda_visible_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
    os.chdir(curr_dir)
    
    return os.path.abspath(model_path)
    

def launch_lora_merge(model_path, args):
    final_model_path = os.path.abspath(
        os.path.join(os.path.dirname(model_path), os.path.basename(model_path) + "_merged")
    )

    if args.eval_only:
        return final_model_path
    curr_dir = os.getcwd()
    os.chdir(OPENINSTRUCT_DIR)

    merge_lora_command = [
        "python", "open_instruct/merge_lora.py",
        "--lora_model_name_or_path", model_path,
        "--output_dir", final_model_path,
        # "--pad_to_multiple_of", "8",
        "--save_tokenizer"
    ]

    subprocess.run(merge_lora_command, env=os.environ, stdout=sys.stdout, stderr=subprocess.STDOUT)
    os.chdir(curr_dir)

    return final_model_path


def launch_evals(args, model_path, result_path):
    curr_dir = os.getcwd()
    os.chdir(OPENINSTRUCT_DIR)
    eval_command = [
        "python", EVAL_SCRIPT_PATH,
        "--model_name", model_path,
        "--results_name", result_path,
        "--eval_list", *args.eval_list,
        "--results_base_dir", RESULTS_DIR
    ]
    if args.skip_eval_if_exists:
        eval_command.append("--skip_if_exists")
    print(f"Running command: {' '.join(eval_command)}")
    subprocess.run(eval_command, env=os.environ, stdout=sys.stdout, stderr=subprocess.STDOUT)
    os.chdir(curr_dir)



def main(args):
    ######################################################## Check that we have enough gpus and set up ############################
    avail_gpus = torch.cuda.device_count()
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', ",".join(map(str, range(avail_gpus))))

    if avail_gpus < args.num_gpus:
        raise ValueError(f"Number of GPUs requested ({args.num_gpus}) is greater than available GPUs ({avail_gpus})")
    elif avail_gpus > args.num_gpus:
        cuda_visible_devices = ",".join(cuda_visible_devices.split(",")[:args.num_gpus])
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

    ######################################################## load training config ############################################
    
    with open(args.train_yaml_path, 'r') as f:
        train_config = yaml.safe_load(f) 

    ######################################################## launch training ########################################################

    starttime = time.time()
    model_path = launch_training_pipeline(args, train_config)
    print(f"Training took {time.time() - starttime} seconds")

    ######################################################## merge lora if necessary and set model_path ########################################################
    use_lora = train_config.get("lora_rank", 0) > 0
    
    if use_lora:
        print("Merging Lora")
        starttime = time.time()
        final_model_path = launch_lora_merge(model_path, args)
        print(f"Lora merge took {time.time() - starttime} seconds")
    else:
        final_model_path = model_path

    ######################################################## run evals ########################################################

    # results_name = os.path.relpath(model_path, MODEL_OUTPUT_DIR)
    # results_name = train_config.get('results_name', results_name)
    # starttime = time.time()
    # launch_evals(args, final_model_path, results_name)
    # print(f"Evals took {time.time() - starttime} seconds")




if __name__ == "__main__":
    args = parse_args()
    if os.path.isdir(args.train_yaml_path):
        all_train_yamls = [os.path.join(args.train_yaml_path, f) for f in os.listdir(args.train_yaml_path) if f.endswith('.yaml')]
    else:
        all_train_yamls = [args.train_yaml_path]
    
    print(f"Found {len(all_train_yamls)} training configs: {all_train_yamls}")
    
    for train_yaml_path in all_train_yamls:
        args.train_yaml_path = train_yaml_path
        main(args)
