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
OUTPUTS_DIR = "/home/stellalisy/health-q/data/alpaca_eval/model_generations"

GENERATION_COMMAND = "openrlhf.cli.batch_inference"

def parse_args():
    parser = argparse.ArgumentParser(description="Launch generation pipeline with specified configuration.")
    parser.add_argument('--master_port', type=int, default=29500, help='Master port for distributed generation')
    parser.add_argument('--deepspeed_stage', type=int, choices=[0, 1, 2, 3], default=None, help='DeepSpeed stage')
    parser.add_argument('--num_gpus', type=int, default=8, help='Number of GPUs to use')
    parser.add_argument('--generation_yaml_path', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--generation_overrides', type=str, default=None, help='Overrides for the generation config')
    parser.add_argument('--eval_list', type=str, nargs='+', default=["gsm", "math", "ifeval", "bbh", "truthfulqa", "alpaca"], help='List of evaluations to run')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluations')
    parser.add_argument('--skip_eval_if_exists', action='store_true', help='Skip evaluations on datasets where the results already exist')

    args = parser.parse_args()
    return args

def launch_generation_pipeline(args, config):
    output_path = os.path.join(OUTPUTS_DIR, config['output_path'])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
 
    curr_dir = os.getcwd()
    os.chdir(OPENRLHF_DIR)

    print(f"Generating completions using {args.num_gpus} GPUs")
    print(f"Generation config: {args.generation_yaml_path}")
    print(f"DeepSpeed stage: {args.deepspeed_stage}")

    cuda_visible_devices = None
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        print(f"Using CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
        del os.environ['CUDA_VISIBLE_DEVICES']

    if args.deepspeed_stage is not None:
        config["zero_stage"] = args.deepspeed_stage

    if args.generation_overrides is not None:
        for override in args.generation_overrides.split(","):
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
    
    generation_args = []
    for k, v in config.items():
        # Skip the keys that have false values
        if v is False:
            continue
        generation_args.append(f"--{k}")
        # if its a --store_true argument, we don't need to add the value
        if v is True or v is None:
            continue
        # otherwise, add the value
        generation_args.append(str(v))

    generation_command = ['deepspeed', '--master_port', str(args.master_port)]
    if cuda_visible_devices is not None:
        generation_command.extend(['--include', 'localhost:' + cuda_visible_devices])
    
    generation_command += ['--module', GENERATION_COMMAND]
    generation_command += generation_args

    print(f"Running command: {' '.join(generation_command)}")
    subprocess.run(generation_command, env=os.environ, stdout=sys.stdout, stderr=subprocess.STDOUT)
    
    if cuda_visible_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
    os.chdir(curr_dir)
    
    return os.path.abspath(output_path)



def main(args):
    ######################################################## Check that we have enough gpus and set up ############################
    avail_gpus = torch.cuda.device_count()
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', ",".join(map(str, range(avail_gpus))))

    if avail_gpus < args.num_gpus:
        raise ValueError(f"Number of GPUs requested ({args.num_gpus}) is greater than available GPUs ({avail_gpus})")
    elif avail_gpus > args.num_gpus:
        cuda_visible_devices = ",".join(cuda_visible_devices.split(",")[:args.num_gpus])
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

    ######################################################## load generation config ############################################
    
    with open(args.generation_yaml_path, 'r') as f:
        generation_config = yaml.safe_load(f) 

    ######################################################## launch generation ########################################################

    starttime = time.time()
    output_path = launch_generation_pipeline(args, generation_config)
    print(f"Generation took {time.time() - starttime} seconds, saved to {output_path}")




if __name__ == "__main__":
    args = parse_args()
    if os.path.isdir(args.generation_yaml_path):
        all_generation_yamls = [os.path.join(args.generation_yaml_path, f) for f in os.listdir(args.generation_yaml_path) if f.endswith('.yaml')]
    else:
        all_generation_yamls = [args.generation_yaml_path]
    
    print(f"Found {len(all_generation_yamls)} generation configs: {all_generation_yamls}")
    
    for generation_yaml_path in all_generation_yamls:
        args.generation_yaml_path = generation_yaml_path
        main(args)
