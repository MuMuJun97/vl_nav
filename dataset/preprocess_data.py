import copy
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
      "short": "Please help me navigate inside the building. "
               "I will provide 12 images of the environment from different direction angles.",
    },
    ############## SOON ##############
    'soon_target': [
        "what does the {target} look like?",  # 0
        "where is the {target}?",  # 1
        "which room or area is the {target} in?",  # 2
        "how to find the {target}?",  # 3
    ],
    ###########################################
    # 'soon_target': [
    #     "what does the {target} look like?",  # 0
    #     "what is the relationship between the {target} and other objects in the room?",  # 1
    #     "which room or area is the current location in?",  # 2
    #     "what is the relationship between the current room and other neighboring rooms?",  # 3
    # ],
    ############## Fine-grained R2R ##############
    'R2R': {
        'instr2view':{
            'question': "which direction does the navigation instruction \"{Instruction}\" refer to?",
            'answer'  : "{ViewID}",
        },
        'view2instr':{
            'question': "how to get to direction {ViewID}?",
            'answer'  : "{Instruction}",
            'stop_question': "how to stop?",
        },
        'stop':{
            'question': "what is the next action for the navigation instruction \"{Instruction}\"?",
            'answer'  : "stop",
        },

    },
    # 'R2R': [
    #     "which direction does the navigation instruction \"{Instruction}\" refer to?", # 0: instr2view
    #     "how to get to direction {ViewID}?", # view2instr
    # ],
    # 'R2RAnswer': [
    #     "{ViewID}",
    #     "{Instruction}",
    # ],
    ###########################################
    # 'R2R': [
    #     "what is the next step I should take based on the instruction: {Instruction}?",
    #     "I am going to direction {ViewID}, what do I do?",
    # ],
    # 'R2RAnswer': [
    #     "you should go in direction {ViewID}.",
    #     "you can follow the navigation instruction: {Instruction}",
    # ],

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

def preprocess_soon_v1(soon_file, navigable_loc):
    assert soon_file.exists()
    with open(str(soon_file), "r") as f:
        soon_data = json.load(f)
    res_data = []

    item_idx = 0
    pbar = tqdm(soon_data, desc="preprocess soon data:")
    for idx, _ in enumerate(pbar):
        for path in soon_data[idx]['path'][:len(promptQAs['soon_target'])]:
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

        # print(question_text)
        # print(answer)
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

def save_question_answer(res_data, type="fr2r"):
    """
    @func:
        save SOON dataset question and answer to json.
    """
    all_qas = dict()
    for idx, _item in enumerate(res_data):
        all_qas[idx] = dict()
        if type == "fr2r":
            all_qas[idx]['question_instr2view'] = _item['qa']['question_instr2view']
            all_qas[idx]['answer_instr2view'] = _item['qa']['answer_instr2view']
            all_qas[idx]['question_view2instr'] = _item['qa']['question_view2instr']
            all_qas[idx]['answer_view2instr'] = _item['qa']['answer_view2instr']
        else:
            all_qas[idx]['question'] = _item['qa']['question']
            all_qas[idx]['answer'] = _item['qa']['answer']
    qa_file = "data/{}_qa.json".format(type)
    with open(str(qa_file), 'w') as f:
        json.dump(all_qas, f, indent=2)


def preprocess_fr2r(fr2r_file, navigable_loc):
    assert fr2r_file.exists()
    with open(str(fr2r_file),"r") as f:
        fr2r_data = json.load(f)
    res_data = []
    item_idx = 0
    pbar = tqdm(fr2r_data, desc="preprocess fine-grained data:")

    stop_cases = []
    stop_sum = 0
    filter_cases_path = 0
    filter_cases_instr = 0

    answers_type = {
        'stop':0,
        'all':0
    }
    for i in range(12):
        answers_type[i] = 0

    enable_instr2view = True
    enable_view2instr = True

    for idx, _ in enumerate(pbar):
        for j,chunk in enumerate(fr2r_data[idx]['chunk_view']):
            for k,sub_path in enumerate(chunk):
                item = dict()
                item['scan'] = fr2r_data[idx]['scan']
                item['fr2r'] = {
                    'distance': fr2r_data[idx]['distance'],
                    'path_id': fr2r_data[idx]['path_id'],
                    'heading': fr2r_data[idx]['heading'],
                }

                # navigable_pathViewIds = [0] # start_view=0
                navigable_pathViewIds = []
                for curIx in range(len(fr2r_data[idx]['path'])-1):
                    curNode = fr2r_data[idx]['path'][curIx]
                    nextNode = fr2r_data[idx]['path'][curIx+1]
                    nextViewId = navigable_loc[item['scan']][curNode][nextNode]['pointId']
                    navigable_pathViewIds.append(nextViewId)
                navigable_pathViewIds.append(-1) # end_view=-1
                # remove shorter sub-path;
                start_index = sub_path[0]-1
                if sub_path[1] != len(fr2r_data[idx]['path']):
                    end_index = sub_path[1]-1
                else:
                    end_index = sub_path[1]
                if end_index - start_index < 1:
                    filter_cases_path += 1
                    continue

                item['path'] = fr2r_data[idx]['path'][start_index:end_index]
                new_instructions = ast.literal_eval(fr2r_data[idx]['new_instructions'])

                cur_sub_instr = new_instructions[j][k]

                ### ['and', 'turn', 'left'] --> ['turn', 'left']
                ### ['and', 'stop'] --> ['stop']
                if 'and' in cur_sub_instr[0]:
                    cur_sub_instr = cur_sub_instr[1:]

                ### remove short cases: ['stop']
                if len(cur_sub_instr) <= 2:
                    filter_cases_instr += 1
                    continue

                # TODO: set next direction ID to current viewpoint
                item['navigable_pathViewIds'] = navigable_pathViewIds[start_index:end_index]

                if len(item['navigable_pathViewIds']) < 1:
                    continue

                # # remove cases: only STOP action
                # if item['navigable_pathViewIds'][-1] == -1 and len(item['navigable_pathViewIds']) < 2:
                #     continue

                item['qa'] = {
                    'full_instr': fr2r_data[idx]['instructions'][j],
                    'sub_instr': " ".join(cur_sub_instr)
                }

                if 'stop' in item['qa']['sub_instr']:
                    stop_cases.append(item['qa']['sub_instr'])
                stop_sum += 1

                item_instr2view = copy.deepcopy(item)
                item_view2instr = copy.deepcopy(item)

                ###### [1] qa type: give instruction, ask direction ViewID ######
                # "which direction does the navigation instruction \"{Instruction}\" refer to?"
                if enable_instr2view:
                    question_instr2view = "Question:{}".format(
                        promptQAs['R2R']['instr2view']['question']
                    )
                    question_instr2view = question_instr2view.format(
                        Instruction=item['qa']['sub_instr']
                    )

                    # query next step:
                    if item['navigable_pathViewIds'][-1] == -1: # next step: STOP
                        prob = random.random()
                        if prob < 0.2:
                            vp_index = random.randint(0, len(item['path']) - 1)
                        elif len(item['path']) == 1:
                            vp_index = random.randint(0, len(item['path']) - 1)
                        elif len(item['path']) > 1:
                            # not consider STOP.
                            vp_index = random.randint(0, len(item['path']) - 2)
                    else:
                        vp_index = random.randint(0, len(item['path']) - 1)
                    ViewpointNext = item['navigable_pathViewIds'][vp_index]  # next direction {0..11}, -1 means STOP

                    if ViewpointNext == -1:
                        answers_type['stop'] += 1
                        answer_instr2view = "Answer:{}".format(promptQAs['R2R']['instr2view']['answer'])
                        answer_instr2view = answer_instr2view.format(ViewID='stop')
                    else:
                        answers_type[ViewpointNext] += 1
                        answer_instr2view = "Answer:{}".format(promptQAs['R2R']['instr2view']['answer'])
                        answer_instr2view = answer_instr2view.format(ViewID=ViewpointNext)
                    answers_type['all'] += 1

                    viewpoint = item['path'][vp_index]

                    item_instr2view['qa']['question'] = question_instr2view
                    item_instr2view['qa']['answer'] = answer_instr2view
                    item_instr2view['viewpoint'] = viewpoint
                    item_instr2view['ViewpointNext'] = ViewpointNext

                ###### [2] qa type: give ViewID, ask instruction ######
                # "how to get to direction {ViewID}?"
                if enable_view2instr:
                    # query next step:
                    if item['navigable_pathViewIds'][-1] == -1: # next step: STOP
                        prob = random.random()
                        if prob < 0.2:
                            vp_index = random.randint(0, len(item['path']) - 1)
                        elif len(item['path']) == 1:
                            vp_index = random.randint(0, len(item['path']) - 1)
                        elif len(item['path']) > 1:
                            # not consider STOP.
                            vp_index = random.randint(0, len(item['path']) - 2)
                    else:
                        vp_index = random.randint(0, len(item['path']) - 1)
                    ViewpointNext = item['navigable_pathViewIds'][vp_index]  # next direction {0..11}, -1 means STOP

                    if ViewpointNext == -1:
                        question_view2instr = "Question:{}".format(
                            promptQAs['R2R']['view2instr']['stop_question']
                        )
                    else:
                        question_view2instr = "Question:{}".format(
                            promptQAs['R2R']['view2instr']['question']
                        )
                        question_view2instr = question_view2instr.format(ViewID=ViewpointNext)
                    answer_view2instr = "Answer:{}".format(promptQAs['R2R']['view2instr']['answer'])
                    answer_view2instr = answer_view2instr.format(
                        Instruction=item['qa']['sub_instr']
                    )

                    viewpoint = item['path'][vp_index]

                    item_view2instr['qa']['question'] = question_view2instr
                    item_view2instr['qa']['answer'] = answer_view2instr
                    item_view2instr['viewpoint'] = viewpoint
                    item_view2instr['ViewpointNext'] = ViewpointNext

                # # qa type 3: give instruction, ask next action: STOP
                # question_stop = "Question:{}".format(
                #     promptQAs['R2R']['stop']['question']
                # )
                # question_stop = question_stop.format(
                #     Instruction=item['qa']['sub_instr']
                # )
                # answer_stop = "Answer:{}".format(promptQAs['R2R']['stop']['answer'])
                #
                # item['qa']['question_instr2view'] = question_instr2view
                # item['qa']['answer_instr2view'] = answer_instr2view
                # item['qa']['question_view2instr'] = question_view2instr
                # item['qa']['answer_view2instr'] = answer_view2instr
                # item['qa']['question_stop'] = question_stop
                # item['qa']['answer_stop'] = answer_stop

                item_instr2view['sample_idx'] = item_idx
                res_data.append(item_instr2view)
                item_idx += 1
                item_view2instr['sample_idx'] = item_idx
                res_data.append(item_view2instr)
                item_idx += 1

    # # TEST Visualization Paths
    # from dataset.utils.visualize_mp3d import mp3d_view
    # for item_s in res_data[:100]:
    #     mp3d_view(item_s)

    ans_msg = "[INFO] answers type frequency: \n "
    for ans,ans_nums in answers_type.items():
        if ans == 'all':
            continue
        if ans == 'stop':
            ans_msg += "[{}]: {:.2f}% ({}/{}) .".format(
                ans, (answers_type[ans] * 100 / answers_type['all']), answers_type[ans], answers_type['all']
            )
        elif int(ans) % 3 == 0:
            ans_msg += "[{}]: {:.2f}% ({}/{}) .\n ".format(
                ans,(answers_type[ans]*100/answers_type['all']),answers_type[ans],answers_type['all']
            )
        else:
            ans_msg += "[{}]: {:.2f}% ({}/{}) .".format(
                ans, (answers_type[ans] * 100 / answers_type['all']), answers_type[ans], answers_type['all']
            )
    print(ans_msg)
    print('[INFO] collecting Fine-grained dataset {} = 2 x {} samples'.format(len(res_data),stop_sum))
    print('[INFO] filter and remove: (1) short sub-path {} samples; (2) short sub-instruct {} samples'.format(
        filter_cases_path, filter_cases_instr
    ))
    print('[INFO] there are {:.2f}% ({}/{}) STOP samples in Fine-grained R2R dataset'.format(
        (len(stop_cases)/stop_sum),len(stop_cases),stop_sum
    ))
    return res_data


def generate_direction_from_mp3d(navigable_loc, mode="all"):
    item_idx = 0
    res_data = []

    if mode == "only_connect":
        for scan,scan_dict in navigable_loc.items():
            for viewpoint, viewpoint_dict in scan_dict.items():
                viewpoint_direction_ids = []
                for neighbor, neighbor_dict in viewpoint_dict.items():
                    if neighbor == viewpoint:
                        continue

                    if neighbor_dict['pointId'] in viewpoint_direction_ids:
                        continue
                    else:
                        viewpoint_direction_ids.append(neighbor_dict['pointId'])

                    item = dict()
                    item['sample_idx'] = item_idx
                    item['scan'] = scan
                    item['viewpoint'] = viewpoint
                    item['neighbor'] = neighbor
                    item['directionID'] = neighbor_dict['pointId']

                    # qa type 2: give ViewID, ask instruction
                    question_view2instr = "Question:{}".format(
                        promptQAs['R2R']['view2instr']['question'].format(
                            ViewID=item['directionID']
                        )
                    )

                    item['qa'] = {
                        'question_view2instr': question_view2instr,
                    }
                    res_data.append(item)
                    item_idx += 1

    if mode == "all":
        for scan, scan_dict in navigable_loc.items():
            for viewpoint, viewpoint_dict in scan_dict.items():
                for direction_id in range(12):
                    item = dict()
                    item['sample_idx'] = item_idx
                    item['scan'] = scan
                    item['viewpoint'] = viewpoint
                    item['directionID'] = direction_id

                    # qa type 2: give ViewID, ask instruction
                    question_view2instr = "Question:{}".format(
                        promptQAs['R2R']['view2instr']['question'].format(
                            ViewID=item['directionID']
                        )
                    )

                    item['qa'] = {
                        'question_view2instr': question_view2instr,
                    }
                    res_data.append(item)
                    item_idx += 1
    print('[INFO] select {} samples from mp3d'.format(len(res_data)))
    return res_data

