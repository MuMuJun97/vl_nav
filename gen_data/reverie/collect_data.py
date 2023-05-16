import json
import random
import jsonlines


all_data = []
num = 40
for i in range(num):
    path = 'rewriter_results_processor_{}.jsonl'.format(i)
    print(path)
    with open(path, 'rb') as f:
        for line in f:
            cc = json.loads(line)
            all_data.append(cc)

out_data = []
for data in all_data:
    if not ('Question:' in data[3] and 'Answer:' in data[3]): continue
    cc = {}
    cc['scan'] = data[1]
    cc['path'] = data[2]
    ret = data[3].split('Answer:')
    if not len(ret) == 2: continue
    if not 'Question:' in ret[0]: continue
    ret[0] = ret[0].replace('Question:', "")
    cc['question'] = ret[0].strip()
    cc['answer'] = ret[1].strip()
    out_data.append(cc)

import pdb;pdb.set_trace()
writer_train = jsonlines.open('reverie_qa_v1.jsonl', 'w')
for cc in out_data:
    writer_train.write(cc)
