# import pandas as pd
# import jsonlines
# import threading
import time
import requests
import json
import random


def get_chatgpt_rsp(messages, token):
    headers = {'content-type': "application/json", 'Authorization': f'Bearer {token}'}
    return requests.post(
            "https://api.openai.com/v1/chat/completions",
            json = {
              "model": "gpt-3.5-turbo",
              "messages": messages
            },
            headers=headers
            ).json()


def ask_turbo(question, token):
    messages=[
          {"role": "user", "content": question}
        ]
    return get_chatgpt_rsp(messages, token)




path = '/mnt/lustre/huangshijia.p/MM/vl_nav/data/REVERIE/REVERIE_train.json'
with open(path, 'rb') as f:
  in_data = json.load(f)

all_data = []
idx = 0
for data in in_data:
  cc = {}
  cc['scan'] = data['scan']
  cc['path'] = data['path'][-1]
  for text in data['instructions']:
    cc['instruction'] = text
    cc['id'] = idx
    idx += 1
    all_data.append(cc)

import pdb;pdb.set_trace()


text = 'You are an assistant for data generation. '
text += 'Given a text, generate a question answer pair based on the text.\n'
text += 'Text: Go to the lounge area and clean the top picture above the lamp.\n'
text += 'Question: what object is on the lamp?\n'
text += 'Answer: a picture.\n'
text += 'Text: Go to the lounge room and pick up the top picture above the lamp.\n'
text += 'Question: where are we?\n'
text += 'Answer: we are in the lounge room.\n'
text += 'Text: Go to the laundryroom on level 1 and turn on the exaust fan.\n'
text += 'Question: where is the exaust fan?\n'
text += 'Answer: it is in the laundryroom.\n'
text += 'Text: {}\n'.format(all_data[8]['instructions'][0])
# text += 'Given a question, please output a {}, {} and {} answer A, and a {}, {} but {} answer B.'\
#     .format(option[c1][d1][1], option[c2][d2][1], option[c3][d3][1], option[c1][d1][1], option[c2][d2][1], option[c3][d3][0])
# text += '\nThe output format is "Answer A: {Content}\n Answer B: {Content}". \nQuestion: '
# text += 'How can we reduce air pollution?'
# text += 'Generate some topics related to finance.'
# text += "Name three government positions in the United States."


print(text)
print("==================")
tokens = [
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
"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",
]
token = tokens[2]

rsp = ask_turbo(text, token)
rsp = rsp["choices"][0]["message"]["content"]
print(rsp)
print("==================")
