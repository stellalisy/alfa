template="""

You could use some parsed auxiliary information such as the final diagnosis and conclusion. Make sure that the multiple choice questions you generate are not too easy but also not impossible to answer. Based on this patient record, generate three different medical exam multiple choice questions and store them in the following json format:

{
    "question1": [generated question 1],
    "question1_optionA": [option A],
    "question1_optionB": [option B],
    "question1_optionC": [option C],
    "question1_optionD": [option D],
    "question1_correct_answer": [A or B or C or D],
    "question2": [generated question 2],
    ....
}

After you generate the three questions, do a round of revision. In your revision, you should:
1. Identify any medical inaccuracies in your first response, corrsect them if any exists.
2. Make sure the questions are sufficiently difficult but not impossible to answer with the patient information.
3. The correct answer is indeed correct, if none of the options are correct or more than one options are correct, revise the options to improve the question.
4. Ensure that the correct answers are more evenly distributed among the available options (shuffle if necessary) to enhance the quality and reliability of the questions.
5. Guarantee that the json output is parsable.

Respond with the final revised question in the json format and NOTHING ELSE."""

system_prompt="""You are a experienced expert working in the field of medicine education. Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, you are tasked to create real-world medical entrance exam multiple choice questions for students given a patient case. The generated multiple choice should consist of a question and 4 options, which could be answered by the given patient conversation. Base your response on the current and standard practices referenced in medical guidelines. The created question should be answerable only with the patient information, rather than testing some hardcore scientific foundational knowledge recall. The questions should be challenging but not impossible to answer. The correct answer should be correct, and the distractors should be plausible. The correct answer should be evenly distributed among the available options to enhance the quality and reliability of the questions. The output should be in json format."""