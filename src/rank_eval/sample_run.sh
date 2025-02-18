root_dir='.'
script_dir=${root_dir}/src/rank_eval
config_filepath=${script_dir}/sample_config.yaml
annotator_name=healthQ_cot_gpt4o_cot_clf
self_consistency=1
model_base=llama-3.2-3b # llama-3.1-8b
ref_key=base # human, sft, etc. depends on what the reference key name is in your reference file
if [ "$ref_key" = "base" ]; then
    output_key=output_base_${model_base}
elif [ "$ref_key" = "sft" ]; then
    output_key=output_sft_${model_base}
else
    output_key=output_${ref_key}
fi
annotator_key=ANTT_${annotator_name}_sc${self_consistency}_${ref_key}

python ${script_dir}/run_rank_eval.py \
    --root_dir ${root_dir} \
    --run_config_filepath ${config_filepath} \
    --default_annotator ${annotator_name} \
    --annotator_config_self_consistency ${self_consistency} \
    --annotator_key ${annotator_key} \
    --reference_output_key ${output_key}
