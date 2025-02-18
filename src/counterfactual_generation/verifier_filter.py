import json
from collections import defaultdict

dimensions = ["accuracy", "answerability", "avoidbias", "clarity", "focus", "relevance"]
dimensions = ["overall"]
more_key = "more_future"
less_key = "less"
original_key = "rewrite"

data_dir = "/home/stellalisy/health-q/data/counterfactual_questions/verification"
# data_dir = "/home/stellalisy/health-q/data/counterfactual_questions/verification"
input_validation_filepath = f"{data_dir}/eval_verification_overall.jsonl"
input_train_filepath = f"{data_dir}/train_verification_overall.jsonl"
output_validation_filepath = f"{data_dir}/eval_good_ids_overall.jsonl"
output_train_filepath = f"{data_dir}/train_good_ids_overall.jsonl"

input_filepath = input_validation_filepath
output_filepath = output_validation_filepath

with open(input_filepath, "r") as f:
    data = [json.loads(line) for line in f.readlines()]
    data = {sample['id']: sample for sample in data}
corrects = {}
good_ids = []
for dimension in dimensions:
    corrects[dimension] = defaultdict(list)
    for qid, sample in data.items():
        model_ranking = sample[dimension]['rankings'][dimension]['ranking']
        more_index = model_ranking.index(more_key) if more_key in model_ranking else -1
        less_index = model_ranking.index(less_key) if less_key in model_ranking else -1
        original_index = model_ranking.index(original_key) if original_key in model_ranking else -1

        # compare more > less
        if more_index>-1 and less_index>-1 and more_index < less_index and '>' in model_ranking[more_index+len(more_key):][:less_index-more_index-len(more_key)]:
            corrects[dimension]["ML"].append(True)
            good_ids.append(f"{qid}_{dimension}_{more_key}_{less_key}")
        else:
            corrects[dimension]["ML"].append(False)
        
        # compare more > original
        if more_index>-1 and original_index>-1 and more_index < original_index and '>' in model_ranking[more_index+len(more_key):][:original_index-more_index-len(more_key)]:
            corrects[dimension]["MO"].append(True)
            good_ids.append(f"{qid}_{dimension}_{more_key}_{original_key}")
        else:
            corrects[dimension]["MO"].append(False)

        # compare original > less
        if less_index>-1 and original_index>-1 and original_index < less_index and '>' in model_ranking[original_index+len(original_key):][:less_index-original_index-len(original_key)]:
            corrects[dimension]["OL"].append(True)
            good_ids.append(f"{qid}_{dimension}_{original_key}_{less_key}")
        else:
            corrects[dimension]["OL"].append(False)

for dimension in dimensions:
    for pair in ['ML', 'MO', 'OL']:
        accuracy = sum(corrects[dimension][pair])/len(corrects[dimension][pair])
        print(f"{dimension} {pair} acc={accuracy}")
    accuracy = sum(corrects[dimension]['ML']+corrects[dimension]['MO']+corrects[dimension]['OL'])/len(corrects[dimension]['ML']+corrects[dimension]['MO']+corrects[dimension]['OL'])
    print(f"{dimension} acc={accuracy}")

with open(output_filepath, "w") as f:
    for good_id in good_ids:
        f.write(f"{good_id}\n")