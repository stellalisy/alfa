group_key=test
exp_key=alfa_3b_dpo_datamix
model_path=stellalisy/alfa_3b_dpo_datamix
root_dir=src/mediq_eval
data_dir=data/mediq_eval


python ${root_dir}/src/mediQ_benchmark.py \
    --healthQ_eval --patient_class FactSelectPatient \
    --expert_class ScaleExpert --max_questions 15 \
    --rationale_generation \
    --expert_model_question_generator ${model_path} \
    --data_dir ${data_dir} \
    --dev_filename medqa_test_convo.jsonl \
    --output_filename ${data_dir}/output/${group_key}/${exp_key}.jsonl \
    --log_filename ${root_dir}/results/${group_key}/${exp_key}.log \
    --history_log_filename ${root_dir}/log/mediq_eval/${group_key}/${exp_key}_history.log \
    --detail_log_filename ${root_dir}/log/mediq_eval/${group_key}/${exp_key}_detail.log \
    --message_log_filename ${root_dir}/log/mediq_eval/${group_key}/${exp_key}_message.log \
    --healthQ_no_system_prompt
