import json
import os
import re
import numpy as np
instr_path = "/home/zlin/vln/llm/vl_nav/build/data/train_10_pred_20230417-223427.json"
with open(str(instr_path), "r") as f:
    instr_data = json.load(f)

question_dict = dict()
for i in range(12):
    question_dict[i] = 0
question_dict.update({
    'stop':0,
    'all':0
})

for idx,item in instr_data.items():
    question = item['_gt_question_text']
    answer = item['_gt_answer_text']

    # 'Question:which direction does the navigation instruction "wait near the bench" refer to?'
    if 'which direction does the navigation instruction' in question:
        question_dict['all'] += 1
        if 'stop' in answer:
            question_dict['stop'] += 1
        else:
            view_id = list(map(int, re.findall('\d+', answer)))[0]
            question_dict[view_id] += 1

for k,v in question_dict.items():
    if k == 'all':
        continue
    print('question {} : {:.2f}% ({}/{})'.format(
        k, v/question_dict['all']*100, v, question_dict['all']
    ))