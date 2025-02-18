import json

split = "test"
filepath = f"/fsx-sage/stellalisy/social_alignment/askdocs/data/ids/{split}_q.jsonl"

with open(filepath, 'r') as f:
    data = [json.loads(line)["qid"] for line in f]

with open(f"/fsx-sage/stellalisy/social_alignment/askdocs/data/ids/{split}_qids.txt", 'w') as f:
    for qid in data:
        f.write(f"{qid}\n")