import json

questions_filepath = "/home/stellalisy/health-q/data/ids/test_q.jsonl"
qids_filepath = "/home/stellalisy/health-q/data/ids/test_qids.txt"

with open(questions_filepath, "r") as f:
    questions = [json.loads(line.strip()) for line in f.readlines()]
with open(qids_filepath, "r") as f:
    qids = [line.strip() for line in f.readlines()]

expert_qids = []
for i, question in enumerate(questions):
    if question["comment_author_flair_text"] != None and "not" not in question["comment_author_flair_text"]:
        expert_qids.append(qids[i])

with open("/home/stellalisy/health-q/data/ids/test_q_expert_qids.txt", "w") as f:
    for qid in expert_qids:
        f.write(f"{qid}\n")