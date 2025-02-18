root_dir="src/training/"

python ${root_dir}scripts/launch_dpo_with_yaml.py \
    --num_gpus 4 \
    --train_yaml_path ${root_dir}sample_configs/sample_config_dpo.yaml