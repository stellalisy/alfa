root_dir="src/training/"

python ${root_dir}scripts/launch_ppo_with_yaml.py \
    --num_gpus 8 \
    --train_yaml_path ${root_dir}sample_configs/sample_config_ppo.yaml