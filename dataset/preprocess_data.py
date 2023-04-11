import json
from tqdm import tqdm
import random
import ast
from copy import deepcopy
import sys

promptQAs = {
    ############## Task Descriptions ##############
    'task_description': {
      "common": "Imagine you are navigating an indoor building with the help of an embodied agent. "
                "You can ask the agent for its current viewpoint, "
                "and provide a navigation language instruction to receive the next direction you should go. "
                "Additionally, you can ask the agent about any object attributes "
                "and positional relationships in the current room to gain a better understanding of the environment. "
                "The agent will provide 12 discrete heading view images of the current location, "
                "covering a 360-degree panorama, and output the next direction "
                "or whether to stop based on the navigation instruction provided.",
      "simple": "Task: Help navigate an indoor building using an embodied agent. "
                "At each location, you have access to 12 view images covering a 360-degree panorama. "
                "Given a navigation instruction, provide the next direction to take or indicate to stop.",
      "middle": "Task Description: Navigate an indoor building with the help of an embodied agent. "
                "Ask the agent for its current viewpoint and receive the next direction to go "
                "based on a navigation language instruction. "
                "Get information about object attributes and positional relationships in the current room "
                "from the agent's 12 discrete heading view images. "
                "The agent will also provide the option to stop if necessary.",
    },
    ############## SOON ##############
    'soon_target': [
        "what does the {target} look like?",  # 0
        "what is the relationship between the {target} and other objects in the room?",  # 1
        "which room or area is the current location in?",  # 2
        "what is the relationship between the current room and other neighboring rooms?",  # 3
    ],
    ############## Fine-grained R2R ##############
    'R2R': [
        "what is the next step I should take based on the instruction: {Instruction}?",
        "I am going to direction {ViewID}, what do I do?",
    ],
    'R2RAnswer': [
        "you should go in direction {ViewID}.",
        "you can follow the navigation instruction: {Instruction}",
    ],
    ############## Other Options ##############
    ############## Other Options ##############
    ############## Other Options ##############
    'soon_qa': [
        "What are the attributes of the target object?",                                     # 0
        "What is the relationship between the target object and other objects in the room?", # 1
        "Which room or area is the target object in?",                                       # 2
        "What is the relationship between the target room and other neighboring rooms?",     # 3
        "What is the navigation instruction for the current scene?",                         # 4 full instruction
        "What is the target object of navigation?",                                          # 5 target object
    ],
    'image2text': [
        "What direction should I turn after reaching <Image>?",
        "What should I do after reaching <Image>?",
        "What is the next step after reaching <Image>?",
    ],
    'image+text': [
        "Based on the image <Image> and the instruction <Instruction>, what is the next step I should take?",
        "Using the image <Image> and the instruction <Instruction>, what is the next landmark I should look for?",
        "Given the instruction <Instruction>, which direction should I turn based on the image <Image>?",
        "What is the next direction to follow after reaching the location shown in <Image>, given the instruction <Instruction>?",
        "From the current image, which direction should I turn based on the instruction <Instruction>?",
        "Which direction should I turn based on the instruction <Instruction>?", # 5
        "What is the next step I should take based on the instruction: <Instruction>?", # 6
    ],
    'image+viewpoint': [
        "From the current image <Image>, what should I do to reach <Direction-{}>?",
        "After reaching the current image <Image>, what should I do to get to <Direction-{}>?",
        "What is the next step after reaching <Image> and facing <Direction-{}>?",
    ],
    # TODO image-text format, from https://github.com/mlfoundations/open_flamingo
    "open_flamingo": [
        "<image>{text}<|endofchunk|>{tokenizer_eos_token}",

        # <image_0>..<image_11>{text}<|endofchunk|>{tokenizer_eos_token}
        "".join(["<image_{}>".format(i) for i in range(12)]) + "{text}<|endofchunk|>{tokenizer_eos_token}"
    ],
    "answer+viewpoint": [
        "You should walk towards <Direction-{ViewID}>.",
        "You should head towards <Direction-{ViewID}>.",
        "You should go in direction {ViewID}.", # 2
        "You should {STOP}.", # 3
    ],
}

def generate_qa(
        question,
        answer,
        tokenizer_eos_token,
        prompt=promptQAs['open_flamingo'][0]
):
    text = " ".join([question,answer])
    input_text = prompt.format(text=text, tokenizer_eos_token=tokenizer_eos_token)
    return input_text


def preprocess_soon(soon_file, navigable_loc):
    assert soon_file.exists()
    with open(str(soon_file), "r") as f:
        soon_data = json.load(f)
    res_data = []

    item_idx = 0
    pbar = tqdm(soon_data, desc="preprocess soon data:")
    for idx, _ in enumerate(pbar):
        for path in soon_data[idx]['path']:
            for instr in soon_data[idx]['instructions']:
                item = dict()
                item['path'] = path
                item['sample_idx'] = item_idx
                item['instruction'] = deepcopy(instr)
                valid_bbox = []
                for bbox in soon_data[idx]['bboxes']:
                    if bbox['image_id'] == path[-1]:
                        if bbox['obj_name'] is None:
                            bbox['obj_name'] = 'target'
                        valid_bbox.append(bbox)
                item['bbox'] = random.choice(valid_bbox)
                item['scan'] = item['bbox']['scan']
                item['instruction'].insert(-1, item['bbox']['obj_name'])

                # navigable_pathViewIds = [0] # start_view=0
                navigable_pathViewIds = []
                for curIx in range(len(item['path'])-1):
                    curNode = item['path'][curIx]
                    nextNode = item['path'][curIx+1]
                    nextViewId = navigable_loc[item['scan']][curNode][nextNode]['pointId']
                    navigable_pathViewIds.append(nextViewId)
                navigable_pathViewIds.append(-1)  # end_view=-1
                item['navigable_pathViewIds'] = navigable_pathViewIds

                res_data.append(item)
                item_idx += 1
        pbar.update(1)

    # TEST Visualization Paths
    # from dataset.utils.visualize_mp3d import mp3d_view
    # for item_s in res_data[:100]:
    #     mp3d_view(item_s)

    pbar = tqdm(res_data, desc="generate soon qa:")
    for idx, _item in enumerate(pbar):
        # for qa:
        # ridx = random.randint(0, 5) # random generate

        # for soon with target object:
        qa_lens = len(promptQAs['soon_target'])
        ridx = idx % qa_lens
        question_text = "Question:{}".format(
            promptQAs['soon_target'][ridx].format(
                target=_item['bbox']['obj_name']
            )
        )
        answer = "Answer:{}".format(
            _item['instruction'][ridx].format()
        )

        # input_text = generate_qa(
        #     question=question_text,
        #     answer=answer,
        #     tokenizer_eos_token=tokenizer_eos_token
        # )

        res_data[idx]['qa'] = dict()
        res_data[idx]['qa']['question'] = question_text
        res_data[idx]['qa']['answer'] = answer
        res_data[idx]['qa']['full_instr'] = res_data[idx]['instruction'][4]

        pbar.update(1)
    return res_data


def preprocess_fr2r(fr2r_file, navigable_loc):
    # prompt_Option = 0
    assert fr2r_file.exists()
    with open(str(fr2r_file),"r") as f:
        fr2r_data = json.load(f)
    res_data = []
    item_idx = 0
    pbar = tqdm(fr2r_data, desc="preprocess fine-grained data:")
    for idx, _ in enumerate(pbar):
        for j,chunk in enumerate(fr2r_data[idx]['chunk_view']):
            for k,sub_path in enumerate(chunk):
                item = dict()
                item['sample_idx'] = item_idx
                item['scan'] = fr2r_data[idx]['scan']
                item['fr2r'] = {
                    'distance': fr2r_data[idx]['distance'],
                    'path_id': fr2r_data[idx]['path_id'],
                    'heading': fr2r_data[idx]['heading'],
                }
                start_index = sub_path[0]-1
                end_index = sub_path[1]
                item['path'] = fr2r_data[idx]['path'][start_index:end_index]
                new_instructions = ast.literal_eval(fr2r_data[idx]['new_instructions'])

                item['qa'] = {
                    'full_instr': fr2r_data[idx]['instructions'][j],
                    'sub_instr': " ".join(new_instructions[j][k])
                }

                # navigable_pathViewIds = [0] # start_view=0
                navigable_pathViewIds = []
                for curIx in range(len(item['path'])-1):
                    curNode = item['path'][curIx]
                    nextNode = item['path'][curIx+1]
                    nextViewId = navigable_loc[item['scan']][curNode][nextNode]['pointId']
                    navigable_pathViewIds.append(nextViewId)
                navigable_pathViewIds.append(-1) # end_view=-1
                item['navigable_pathViewIds'] = navigable_pathViewIds

                # if prompt_Option == 0: # √
                #     question_text = "Question: {}".format(promptQAs['image+text'][4])
                #     question_text = question_text.replace('<Instruction>', item['qa']['sub_instr'])
                #     question = promptQAs['open_flamingo'][0].format(text=question_text,tokenizer_eos_token=tokenizer_eos_token)
                #     answer = "Answer: {}".format(promptQAs['answer+viewpoint'][0])
                # elif prompt_Option == 1:
                #     question = "Question: {}".format(promptQAs['image+text'][2])
                #     question = question.replace('<Instruction>',item['qa']['sub_instr'])
                #     answer = "Answer: {}".format(promptQAs['answer+viewpoint'][0])
                # else:
                #     NotImplementedError

                qa_idx = item_idx % len(promptQAs['R2R'])
                question_text = "Question:{}".format(
                    promptQAs['R2R'][qa_idx]
                )
                if 'Instruction' in question_text:
                    question_text = question_text.format(
                        Instruction=item['qa']['sub_instr']
                    )
                    answer = "Answer:{}".format(promptQAs['R2RAnswer'][qa_idx])
                elif 'ViewID' in question_text:
                    answer = "Answer:{}".format(
                        promptQAs['R2RAnswer'][qa_idx].format(
                            Instruction=item['qa']['sub_instr']
                        )
                    )
                else:
                    raise NotImplementedError


                # input_text = generate_qa(
                #     question=question_text,
                #     answer=answer,
                #     tokenizer_eos_token=tokenizer_eos_token
                # )

                # item['qa']['input_text'] = input_text
                item['qa']['question'] = question_text
                item['qa']['answer'] = answer

                res_data.append(item)
                item_idx += 1

    # # TEST Visualization Paths
    # from dataset.utils.visualize_mp3d import mp3d_view
    # for item_s in res_data[:100]:
    #     mp3d_view(item_s)
    return res_data
