import json

prompt_str = """# Instruction

You will receive a patient's post, a doctor's response, and a list of questions extracted from the doctor's reply. Your task is to **rewrite** each question so that:

1. They are standalone and coherent when asked directly to the patient.
2. They maintain their original meaning and content.

If the patient provides the same answer to multiple questions and the answer is not `None`, combine those questions to make the conversation more efficient, ensuring the patient's response is kept the same. For combined questions, concatenate the question IDs into a list. If a question remains standalone, keep its ID as an integer.

**Important**: 
* Do not change the content or meaning of the questions. 
* Maintain the original JSON format for your output.

# Input
## Patient post
Title: {title}
Post: {post}

## Doctor reply
{comment_body}

# Extracted questions
{parsed_questions}"""

fewshot_title = "Strange lumps\/bumps on one side of neck and majora labia?"
few_shot_post = "Im feeling extremely stressed out and worried because of all the strange symptoms I've been having. First off I(20) am 95 pounds, 5'2, and Asian. Over the course of the past two months I had a lump\/bump on the right side of my neck where the cervical nodes are at. It started as one bump but then it slowly started growing into a chain where that area is at. There is one bump thats the biggest and it kind of just goes off from there. This isn't like a soft ball sized bump, its a bit bigger than a pea. I went to the doctors and she prescribed me antibiotics that didn't help. I am not currently taking any medication. I was then referred to a surgeon who didn't want to do anything until I went to get a CT scan. The bumps aren't \u201cpainful\u201d its more tender than anything. They don't itch, and they kind of grow and recede in size every once in a while. Then a few days ago I noticed that on my majora labia there were bumps that wasn't there before (I don't think). It was similar bumps to the ones on my neck. They don't hurt,itch, or feel tender either. They're also smaller than pea sized bumps. These aren't \u201cpimple\u201d bumps either. Its under the skin and there is no redness or white  areas. Unless it was felt, I don't think It would even be noticeable.  Its just the fact that they appeared right after my neck ones.  I just scheduled my CT scan appointment, but I'm still freaking out and wondering if anyone else has had this problem before?"
fewshot_reply = "No other symptoms?  no illness, wounds or anything like that in the area?  enlarged lymph nodes are most commonly from a local inflammatory reaction (which is probably why you got put on abx initially)."
fewshot_parsed_questions = {"questions":[{"qid": 0, "question":"No other symptoms?","patient_answer":"No wounds\/cuts or anything at all. I'm not feverish\/sick either."},{"qid": 1, "question":"No illness, wounds or anything like that in the area?","patient_answer":"No wounds\/cuts or anything at all. I'm not feverish\/sick either."}]}
fewshot_assistant_answer = {"questions":[{"qid":[0, 1], "question":"Have you experienced any other symptoms, illness, wounds, or anything unusual in the area of the bumps on your neck or majora labia?", "patient_answer":"No wounds/cuts or anything at all. I'm not feverish/sick either."}]}

few_shot_examples = [
    {
        "role": "user", 
        "content": prompt_str.format(title=fewshot_title, post=few_shot_post, comment_body=fewshot_reply, parsed_questions=json.dumps(fewshot_parsed_questions))
    }, 
    {
        "role": "assistant", 
        "content": json.dumps(fewshot_assistant_answer)
    }
]

system = "You are a helpful medical dialogue assistant helping a clinician communicate with their patients."
user = prompt_str

single_q_user = """# Instruction

You will receive a patient's post, a doctor's response, and a list of questions extracted from the doctor's reply. Your task is to **rewrite** each question so that:

1. They are standalone and coherent when asked directly to the patient.
2. They maintain their original meaning and content.

If the patient provides the same answer to multiple questions and the answer is not `None`, combine those questions to make the conversation more efficient, ensuring the patient's response is kept the same. For combined questions, concatenate the question IDs into a list. If a question remains standalone, keep its ID as an integer.

**Important**: 
* Do not change the content or meaning of the questions. 

# Input
## Patient post
Title: {title}
Post: {post}

## Doctor reply
{comment_body}

# Extracted question
{parsed_question}"""

fewshot_parsed_question = "No other symptoms? No illness, wounds or anything like that in the area?"
fewshot_singl_q_assistant_answer = "Have you experienced any other symptoms, illness, wounds, or anything unusual in the area of the bumps on your neck or labia majora?"

singl_q_few_shot_example = [
    {
        "role": "user", 
        "content": single_q_user.format(title=fewshot_title, post=few_shot_post, comment_body=fewshot_reply, parsed_question=fewshot_parsed_question)
    }, 
    {
        "role": "assistant", 
        "content": fewshot_singl_q_assistant_answer
    }
]