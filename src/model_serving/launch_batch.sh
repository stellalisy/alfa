for port in {7000..7023}; do  

bash src/model_serving/launch_cli.sh $port

root_dir=src

file_content="#!/bin/bash
#SBATCH --job-name=alfa_serve_405b_fp8_${port}
#SBATCH --account=argon
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --chdir=${root_dir}/model_serving
#SBATCH --export=all
#SBATCH --output=${root_dir}/model_serving/slogs/%x-%j.out
#SBATCH --error=${root_dir}/model_serving/slogs/%x-%j.err

source ~/.bashrc
conda activate health_q

cd ${root_dir}/model_serving
bash ${root_dir}/model_serving/vllm_serve.sh meta-llama/Llama-3.1-405B-Instruct-FP8 submit-1:${port} 8
"

printf '%s\n' "$file_content" > serve_405b_fp8_${port}.sbatch
sbatch serve_405b_fp8_${port}.sbatch
rm serve_405b_fp8_${port}.sbatch

done