import json
import random
import pickle
import time
import os
import torch
import numpy as np
import networkx as nx
from pathlib import Path
import cv2
import ast
import jsonlines
debug = True


def load_r2r_data(anno_file, max_instr_len=200):
    # assert anno_file.exists()
    with open(str(anno_file)) as f:
        data = json.load(f)
    new_data = []
    sample_index = 0
    for i, item in enumerate(data):
        # Split multiple instructions into separate entries
        for j, instr in enumerate(item['instructions']):
            new_item = dict(item)
            new_item['raw_idx'] = i
            new_item['sample_idx'] = sample_index
            new_item['instr_id'] = 'r2r_{}_{}'.format(item['path_id'], j)
            new_item['instruction'] = instr
            new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
            del new_item['instructions']
            del new_item['instr_encodings']
            new_item['data_type'] = 'r2r'
            new_data.append(new_item)
            sample_index += 1
    if debug:
        return new_data[:20]
    else:
        return new_data


def load_reverie_data(anno_file):
    # assert anno_file.exists()
    with open(str(anno_file)) as f:
        data = json.load(f)
    new_data = []
    sample_index = 0
    for i, item in enumerate(data):
        # Split multiple instructions into separate entries
        for j, instr in enumerate(item['instructions']):
            new_item = dict(item)

            if 'objId' in item:
                new_item['instr_id'] = '%s_%s_%s_%d' % ('reverie', str(item['path_id']), str(item['objId']), j)
            else:
                new_item['path_id'] = item['id']
                new_item['instr_id'] = '%s_%s_%d' % ('reverie', item['id'], j)
                new_item['objId'] = None

            new_item['sample_idx'] = sample_index
            # new_item['instr_id'] = 'reverie_{}_{}'.format(item['path_id'], sample_index)
            new_item['instruction'] = instr
            del new_item['instructions']
            new_item['data_type'] = 'reverie'

            new_item['raw_idx'] = None
            new_item['instr_encoding'] = None

            new_data.append(new_item)
            sample_index += 1
    if debug:
        return new_data[:20]
    else:
        return new_data


def load_soon_data(anno_file):

    # with open(str(anno_file)) as f:
    #     data = json.load(f)
    data = []
    with jsonlines.open(str(anno_file), 'r') as f:
        for item in f:
            item['end_image_ids'] = [x['image_id'] for x in item['bboxes']]
            item['image_id_to_obj_label'] = {x['image_id']: x.get('pseudo_label', None) for x in item['bboxes']}
            new_bboxes = {}
            for bbox in item['bboxes']:
                new_bboxes[bbox['image_id']] = bbox
            item['bboxes'] = new_bboxes
            data.append(item)

    new_data = []
    sample_index = 0
    for i, item in enumerate(data):
        # for j, path in enumerate(item['path']):
        # Split multiple instructions into separate entries
        for k, instr in enumerate(item['instructions']):
            new_item = dict(item)
            new_item['sample_idx'] = sample_index
            # soon: idx-path_idx-instr_idx
            new_item['instr_id'] = "soon_{}_{}_{}".format(i, item['path_id'], k)

            # current path
            # new_item['path'] = path
            # path = new_item['path']

            # bboxes = []
            # for bbox in item['bboxes']:
            #     if bbox['image_id'] == path[-1]:
            #         bboxes.append(bbox)
            # new_item['bbox'] = random.choice(bboxes)
            # new_item['scan'] = new_item['bbox']['scan']

            # new_item['instruction'] = instr[4]  # full Instruction

            new_item['instruction'] = instr['full']
            # new_item['path_id'] = '{}_{}'.format(i, k)
            new_item['raw_idx'] = None
            new_item['instr_encoding'] = None
            new_item['heading'] = 0.0
            new_item['data_type'] = 'soon'
            del new_item['instructions']
            del new_item['instr_encodings']
            new_data.append(new_item)
            sample_index += 1

    if debug:
        return new_data[:20]
    else:
        return new_data


def load_cvdn_raw(anno_file, path_type='trusted_path'):
    with open(anno_file,"r") as f:
        data = json.load(f)
    new_data = []
    sample_idx = 0
    for i, item in enumerate(data):
        new_item = dict(item)
        new_item['heading'] = None # item['start_pano']['heading']
        new_item['path'] = None

        # Add 'trusted_path' to gt metadata if necessary.
        # if path_type == 'trusted_path':
        #     planner_goal = item['planner_path'][-1]
        #     if planner_goal in item['player_path'][1:]:
        #         new_item['path'] = item['player_path'][:]
        #     else:
        #         new_item['path'] = item['planner_path'][:]
        # else:
        #     raise NotImplementedError

        if len(item['dialog_history']) == 0:
            new_item['instruction'] = "The goal room contains a {target}.\n".format(target=item['target'])
        else:
            new_item['instruction'] = "The goal room contains a {target}.\n".format(target=item['target'])
            sentences = []
            for turn in item['dialog_history']:
                if turn['message'][-1] == '?' or turn['message'][-1] == '.':
                    msg = turn['message']
                else:
                    msg = turn['message'] + "."
                if turn['role'] == 'navigator':
                    sentences.append("Question: " + msg + "\n")
                elif turn['role'] == 'oracle':
                    sentences.append("Answer: " + msg + "\n")
                else:
                    raise NotImplementedError
            sentences = "".join(sentences)
            new_item['instruction'] += sentences
        if new_item['instruction'][-1] == '\n':
            new_item['instruction'] = new_item['instruction'][:-1]
        new_item['path_id'] = item['instr_id']
        new_item['raw_idx'] = None
        new_item['instr_encoding'] = None
        new_item['data_type'] = 'cvdn'
        new_item['sample_idx'] = sample_idx
        new_item['instr_id'] = 'cvdn_{}_{}'.format(sample_idx, item['instr_id'])

        new_data.append(new_item)
        sample_idx += 1
    if debug:
        return new_data[:20]
    else:
        return new_data


def load_fr2r_data(anno_file):
    """
    Args:
        anno_file:

    Returns:
        new_item:
            chunk_view/sub_path: [[1, 3], [3, 4], [4, 6], [6, 6]]
            new_item['instruction']:
                'Walk between the columns and make a sharp turn right. Walk down the steps and stop on the landing. '
            new_item['sub_paths']:
                {'af3af33b0120469c9a00daa0d0b36799': 'walk between the column',
                 '5be145994f974347850a48cecd04cdcd': 'walk between the column',
                 '79aedad1206b4eea9c4b639ea2182eb7': 'make a sharp turn right',
                 '1c91ed40af2246f2b126dd0f661970df': 'walk down the step',
                 '385019f5d018430fa233d483b253076c': 'walk down the step',
                 'fd263d778b534f798d0e1ae48886e5f3': 'stop on the landing'}
    """
    assert anno_file.exists()
    with open(str(anno_file)) as f:
        data = json.load(f)
    new_data = []
    for i, item in enumerate(data):
        # Split multiple instructions into separate entries
        for j, instr in enumerate(item['instructions']):
            new_item = dict(item)
            new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
            new_item['instruction'] = instr
            del new_item['instructions']
            new_item['data_type'] = 'fr2r'

            new_instructions = ast.literal_eval(new_item['new_instructions'])
            new_instrs = new_instructions[j]

            for ni, n_str in enumerate(new_instrs):
                if 'and' in n_str:
                    new_instrs[ni] = new_instrs[ni][1:]
                new_instrs[ni] = " ".join(new_instrs[ni])

            sub_paths = dict()
            chunk_view = item['chunk_view'][j]
            for ci, cv in enumerate(chunk_view):
                for p_i in range(cv[0]-1, cv[1]):
                    sub_paths[new_item['path'][p_i]] = new_instrs[ci]

            new_item['sub_paths'] = sub_paths
            new_data.append(new_item)
    return new_data


def load_eqa_data(anno_file, split='train', vis=False):
    # raise NotImplementedError
    # assert anno_file.exists()
    with open(str(anno_file)) as f:
        data = json.load(f)[split]

    # filter data:
    new_data = []
    qa_set = set()

    sample_index = 0

    for i, item in enumerate(data):
        qa_item = item['question']['question_text'] + \
                  item['question']['answer_text'] + \
                  item['path'][-1] + \
                  item['path'][0]
        if qa_item in qa_set:
            continue
        else:
            qa_set.add(qa_item)

            new_item = dict(item)
            new_item['sample_idx'] = sample_index
            new_item['instr_id'] = item['episode_id']
            new_item['instruction'] = item['question']['question_text']
            new_item['data_type'] = 'eqa'
            question_type = item['question']['question_type']

            question_location = item['question']['question_location']
            if '_' in question_location:
                item['question']['question_location'] = ' '.join(question_location.split('_'))

            if question_type == 'color_room':
                answer_prompt = "The color of the {target} in the {room} is {answer}".format(
                    target=item['question']['question_object'],
                    room=item['question']['question_location'],
                    answer=item['question']['answer_text']
                )
            elif question_type == 'color':
                answer_prompt = "The color of the {target} is {answer}".format(
                    target=item['question']['question_object'],
                    answer=item['question']['answer_text']
                )
            elif question_type == 'location':
                answer_prompt = "The location of the {target} is {answer}".format(
                    target=item['question']['question_object'],
                    answer=item['question']['answer_text']
                )
            else:
                raise NotImplementedError

            new_item['texts'] = {len(new_item['path']) - 1: [answer_prompt]}
            new_data.append(new_item)
            sample_index += 1

    if False:
        img_dir = Path("/media/zlin/2CD830B2D8307C60/Dataset/features/mp3d_raw_images")
        qa_dict = {}
        # new_data = []
        qa_set = set()
        for i, item in enumerate(new_data):
            scan = item['scan']
            path = item['path']

            location = item['question']['question_location']
            target = item['question']['question_object']
            answer = item['question']['answer_text']

            qa_item = item['question']['question_text'] + path[-1] + answer
            if qa_item in qa_set:
                continue

            qa_set.add(qa_item)

            # if qa_dict.get(location, None) is None:
            #     qa_dict[location] = dict()
            # if qa_dict[location].get(target, None) is None:
            #     qa_dict[location][target] = dict()
            # if qa_dict[location][target].get(answer, None) is None:
            #     qa_dict[location][target][answer] = {
            #         'type': item['question']['question_type'],
            #         'scan': scan,
            #         'target': path[-1],
            #     }
            # else:
            #     prev_qa = qa_dict[location][target][answer]
            #     if prev_qa['target'] == path[-1] \
            #             and prev_qa['scan'] == scan \
            #             and prev_qa['type'] == item['question']['question_type']:
            #         continue

            print(item['question'])
            for j, vp in enumerate(path):
                img_file = img_dir / scan / '{}_{}.png'.format(scan, vp)
                assert img_file.exists()
                panoramic_img = cv2.imread(str(img_file))  # BRG
                panoramic_img = np.hsplit(panoramic_img, 12)
                from dataset.utils.visualize_mp3d import show_12_images
                vimgs = show_12_images(
                    panoramic_img
                )
                cv2.imshow('vimg',vimgs)
                cv2.waitKey(0)
            # new_data.append(item)
    return new_data


def load_cvdn_data(anno_file, shortest_paths=None):
    """
    @Dataset Params:
        inst_idx: unique index of this task instance
        scan: unique scan ID of the house
        target: 对话中要找的物体, Hint: The goal room contains a <target>.
        nav_history: List [viewpoints], Navigator在latest question之前已访问过的nodes
        start_pano: the last question所在的位置, 即 nav_history[-1]
        end_panos: The navigation nodes that compose the end region.
        dialog_history: List, Navigator与Oracle的对话,
            - nav_idx: 每轮dialog有对应的nav_idx,为 nav_history List[] 的索引,
                指示当前对话发生在nav_history中哪个node.
            - role: navigator / oracle
            - message: dialog text
        [1] player_path: navigator响应 the latest answer 之后访问的导航nodes.
        [2] planner_path: 为了回答 the most recent question, 向Oracle展示的导航nodes,
                如果 no dialog history, 则为 towards end_panos 的前5个 shortest path steps.
        game_idx: The unique index of the dialog from which this instance was drawn.

    """
    with open(anno_file,"r") as f:
        data = json.load(f)
    new_data = []
    ignore_count = 0
    long_lengths = 0
    index = 0
    for idx, item in enumerate(data):
        if len(item['dialog_history']) == 0:
            ignore_count += 1
            continue
        new_item = dict()
        new_item['scan'] = item['scan']
        new_item['instr_id'] = item['inst_idx']
        new_item['sample_idx'] = index
        new_item['target_object'] = item['target']
        new_item['history_id'] = item['dialog_history'][-1]['nav_idx']

        # planner_goal = item['planner_path'][-1]
        # if planner_goal in item['player_path'][1:]:  # player walked through planner goal (did not start on it)
        #     trusted_path = item['player_path'][:]  # trust the player.
        # else:
        #     trusted_path = item['planner_path'][:]  # trust the planner.
        trusted_path = item['planner_path'][:]  # trust the planner.
        if True: #item['dialog_history'][0]['nav_idx'] != item['dialog_history'][-1]['nav_idx']:
            # multi-turn dialog

            dialog_text = {}
            paths = []
            texts = {}

            # 遍历history dialog,寻找发生过dialog的nodes.
            dialog_nodes = []
            for j, dialog in enumerate(item['dialog_history']):
                # nav_idx = dialog['nav_idx'] - item['dialog_history'][0]['nav_idx']
                nav_idx = dialog['nav_idx']
                if nav_idx not in dialog_text:
                    dialog_text[nav_idx] = []
                    dialog_nodes.append(item['nav_history'][nav_idx])
                dialog_text[nav_idx].append(dialog['message'])

            history_paths = []
            for hi, history_node in enumerate(dialog_nodes[:-1]):
                history_paths += shortest_paths[item['scan']][history_node][dialog_nodes[hi+1]][:-1]

            # add the latest question-answer position/viewpoint
            history_paths.append(
                item['nav_history'][item['dialog_history'][-1]['nav_idx']]
            )

            max_path_lengths = len(history_paths) + len(item['planner_path'])

            if max_path_lengths > 20:
                continue
                long_lengths += 1
                paths += trusted_path
                all_dialog_text = []
                for k,v in dialog_text.items():
                    for s_txt in v:
                        all_dialog_text.append(s_txt)
                instruction = {
                    0: "".join(all_dialog_text)
                }
            else:
                # get history paths, from the 0-th dialog to the latest dialog
                # paths += # item['nav_history'][item['dialog_history'][0]['nav_idx']:]
                paths += history_paths
                assert paths[-1] == item['player_path'][0]

                if paths[-1] != item['planner_path'][0]:
                    # planner_path是指示给 Oracle 用于回答 most recent question 的 navigation nodes. [Oracle Navigator]
                    # player_path是navigator在得到latest answer后做出的响应. [Human Response]
                    # paths += item['player_path'][1:]
                    # paths[-1] is the current position, item['planner_path'][-1] is the goal location.
                    paths += shortest_paths[item['scan']][
                        paths[-1]
                    ][item['planner_path'][-1]][1:]
                else:
                    assert paths[-1] == trusted_path[0]
                    paths += trusted_path[1:]

                assert paths[0] == dialog_nodes[0]
                for vp, (k, v) in zip(dialog_nodes, dialog_text.items()):
                    texts[paths.index(vp)] = v

        # else:
        #     # single viewpoint: dialog
        #     dialog_text = []
        #     paths = []
        #     for j, dialog in enumerate(item['dialog_history']):
        #         dialog_text.append(
        #             "#{}: {}.\n".format(
        #                 "Question" if dialog['role'] == 'navigator' else "Answer",
        #                 dialog['message']
        #             )
        #         )
        #     dialog_text = "".join(dialog_text)
        #     # check the start position == dialog position
        #     assert item['nav_history'][item['dialog_history'][0]['nav_idx']] \
        #            == item['start_pano']['pano']
        #     assert trusted_path[0] == item['start_pano']['pano']
        #     paths += trusted_path
        #     # the 0-th viewpoint -> dialog_text
        #     instruction = {
        #         0: dialog_text,
        #     }

        new_item['instruction'] = item['target']

        new_item['texts'] = texts
        new_item['path'] = paths

        assert len(paths) < 20

        new_item['data_type'] = 'cvdn'
        new_data.append(new_item)
        index += 1

    # ignore_count=1096: filter 23.11% (1096/4742) samples if there is no dialog history
    # long_lengths=985: filter 20.77% (985/4742)
    return new_data


def generate_data_indexs(data):
    start_index = 0
    end_index = 0
    alldata = []
    all_index = dict()
    if 'r2r' in data.keys():
        alldata += data['r2r']
        end_index += len(data['r2r'])
        all_index.update({i: 'r2r' for i in range(end_index)})
        start_index += len(data['r2r'])
    if 'reverie' in data.keys():
        alldata += data['reverie']
        end_index += len(data['reverie'])
        all_index.update({i: 'reverie' for i in range(start_index, end_index)})
        start_index += len(data['reverie'])
    if 'soon' in data.keys():
        alldata += data['soon']
        end_index += len(data['soon'])
        all_index.update({i: 'soon' for i in range(start_index, end_index)})
        start_index += len(data['soon'])
    if 'fr2r' in data.keys():
        alldata += data['fr2r']
        end_index += len(data['fr2r'])
        all_index.update({i: 'fr2r' for i in range(start_index, end_index)})
        start_index += len(data['fr2r'])
    if 'eqa' in data.keys():
        alldata += data['eqa']
        end_index += len(data['eqa'])
        all_index.update({i: 'eqa' for i in range(start_index, end_index)})
        start_index += len(data['eqa'])
    if 'cvdn' in data.keys():
        alldata += data['cvdn']
        end_index += len(data['cvdn'])
        all_index.update({i: 'cvdn' for i in range(start_index, end_index)})
    return alldata, all_index


def load_nav_graphs(connectivity_dir, scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3] - pose2['pose'][3]) ** 2 \
                + (pose1['pose'][7] - pose2['pose'][7]) ** 2 \
                + (pose1['pose'][11] - pose2['pose'][11]) ** 2) ** 0.5

    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i, item in enumerate(data):
                if item['included']:
                    for j, conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'], data[j]['image_id'], weight=distance(item, data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs


def save_graphs(graph_dict_file_, connectivity_dir_, scans_, logger_):
    graphs_ = load_nav_graphs(connectivity_dir_, scans_)
    shortest_paths_ = {}
    for scan, G in graphs_.items():  # compute all shortest paths
        shortest_paths_[scan] = dict(nx.all_pairs_dijkstra_path(G))
    shortest_distances = {}
    for scan, G in graphs_.items():  # compute all shortest paths
        shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
    graph_dict_ = {'graphs': graphs_, 'shortest_paths': shortest_paths_,
                   'shortest_distances': shortest_distances, 'scans': scans_}
    graph_dict_file_.parent.mkdir(parents=True, exist_ok=True)
    with open(str(graph_dict_file_), "wb") as f:
        pickle.dump(graph_dict_, f)
    logger_.info('Save graph dict to: {}'.format(graph_dict_file_)) if logger_ is not None else None
    return graphs_, shortest_paths_, shortest_distances


def generate_graphs(graph_dict_file, rank, logger, connectivity_dir, scans):
    if rank != 0:
        while not graph_dict_file.exists():
            time.sleep(1)
        with open(str(graph_dict_file), "rb") as f:
            graph_dict = pickle.load(f)
            logger.info('Load graph dict: {}'.format(graph_dict_file)) if logger is not None else None
        graphs = graph_dict['graphs']
        shortest_paths = graph_dict['shortest_paths']
        shortest_distances = graph_dict['shortest_distances']
        del graph_dict
    else:
        if graph_dict_file.exists():
            with open(str(graph_dict_file), "rb") as f:
                graph_dict = pickle.load(f)
                logger.info('Load graph dict: {}'.format(graph_dict_file)) if logger is not None else None
            scans_ = graph_dict.get('scans', None)
            if scans_ is None or sorted(scans_) != sorted(scans):
                graphs, shortest_paths, shortest_distances = save_graphs(
                    graph_dict_file,
                    connectivity_dir,
                    scans,
                    logger
                )
            else:
                graphs = graph_dict['graphs']
                shortest_paths = graph_dict['shortest_paths']
                shortest_distances = graph_dict['shortest_distances']
            del graph_dict
        else:
            graphs, shortest_paths, shortest_distances = save_graphs(
                graph_dict_file,
                connectivity_dir,
                scans,
                logger
            )
    return None, shortest_paths, shortest_distances


###################### Dataset: Load Multi-Step Vision-Language Data ######################
def batch_process_text(batch_dict, tokenizer, max_length, args, image_mask, extract_candidates=None):
    """
    Args:
        batch_dict:
        tokenizer:
        max_length: 512
        args:
        image_mask:
        extract_candidates:
    Returns:
        input_ids:
        attention_mask:
        labels:
    """
    batch_size = batch_dict['batch_size']

    batch_text = tokenizer(
        batch_dict['input_text'],
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False
    )

    batch_gt_text = tokenizer(
        batch_dict['gt_text'],
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False
    )

    input_ids = batch_text['input_ids']
    gt_ids = batch_gt_text['input_ids']
    media_locations = (input_ids >= args.image_token_ids[0]) & (input_ids <= args.image_token_ids[-1])
    media_nums = media_locations.sum(dim=-1) # B

    for bs in range(batch_size):
        media_nums = media_nums[bs].sum()
        image_mask[bs, media_nums:] = False

        # sample_text = tokenizer(
        #     batch_dict['input_text'][bs],
        #     return_tensors="pt",
        #     add_special_tokens=False
        # )
        # sample_text_length = sample_text['input_ids'].shape[-1]
        # # assert sample_text_length < max_length
        # if sample_text_length >= max_length:
        #     cc = (batch_text['input_ids'][bs] >= args.image_token_ids[0]) & \
        #                       (batch_text['input_ids'][bs] <= args.image_token_ids[-1])
        #     media_nums = cc.sum()
        #     image_mask[bs, media_nums:] = False

    action_space_len = len(args.action_token_ids)

    input_ids = batch_text['input_ids']
    labels = torch.zeros_like(input_ids) - 100

    if extract_candidates is None:
        # test: not compute labels, train/val: compute labels
        for bs in range(batch_size):
            sample_text = input_ids[bs]
            # Task-description contains <walkto0>...<walkto11><stop>, remove!
            start_locs = torch.nonzero(sample_text == 1,as_tuple=True)[0].tolist()
            end_locs = torch.nonzero(sample_text == 2,as_tuple=True)[0].tolist()
            if len(end_locs) < len(start_locs):
                end_locs.append(len(sample_text) - 1)
            assert len(start_locs) == len(end_locs)
            for loc_a, loc_b in zip(start_locs, end_locs):
                labels[bs, loc_a+1: loc_b+1] = input_ids[bs, loc_a+1: loc_b+1]
            # only compute loss on: #Answer:<walkto{view_id}>
            # labels[bs,answer_locs] = input_ids[bs,answer_locs]

        return input_ids, batch_text['attention_mask'], labels, image_mask
    else:
        # generate: find the candidate action set
        assert batch_size == 1

        if len(media_locations.shape) == 1:
            media_locs = torch.where(media_locations == True)[0]
        else:
            media_locs = torch.where(media_locations == True)[1]
        loc_diff = torch.diff(media_locs)
        st_loc = torch.where(loc_diff > 1)[0]
        if len(st_loc) == 0:
            st_loc = media_locs[0]
        else:
            st_loc =  st_loc[-1] + 1
            st_loc = media_locs[st_loc]
        end_loc = media_locs[-1]
        # <image0> + 12 --> <walkto0>
        candidates = input_ids[0, st_loc:end_loc+1] + len(args.image_token_ids)

        # add <stop>
        stop_token_id = tokenizer("<stop>",add_special_tokens=False)['input_ids'][-1]
        candidates = torch.cat([candidates, torch.tensor([stop_token_id])])

        return input_ids, batch_text['attention_mask'], labels, image_mask, candidates


def batch_process_image(batch_image, batch_size, batch_angle_feats):
    """
    Args:
        batch_image: List[]
        batch_size:
        batch_angle_feats:

    Returns:

    """
    input_image = []
    input_angle_feats = []
    image_lens = [len(im) for im in batch_image]
    image_mask = torch.arange(max(image_lens)).unsqueeze(0).repeat(batch_size, 1)
    image_mask = image_mask < torch.tensor(image_lens).unsqueeze(1)
    for bs in range(batch_size):
        if image_mask[bs].all():
            # M: multi steps. size: (M*12, 3, 224, 224)
            # image = torch.cat(batch_image[bs], dim=0)
            image = batch_image[bs]

            # angle
            angle_feats = batch_angle_feats[bs]
        else:
            pad_image = torch.zeros_like(batch_image[bs][0]).unsqueeze(dim=0)
            # image = batch_image[bs] + [pad_image] * ((image_mask[bs] == False).sum().item())
            pad_image = [batch_image[bs]] + [pad_image] * ((image_mask[bs] == False).sum().item())
            image = torch.cat(pad_image, dim=0)
            # image = torch.cat(image, dim=0)

            # angle
            pad_angle_feats = torch.zeros_like(batch_angle_feats[bs][0]).unsqueeze(dim=0)
            pad_angle_feats = [batch_angle_feats[bs]] + [pad_angle_feats] * ((image_mask[bs] == False).sum().item())
            angle_feats = torch.cat(pad_angle_feats, dim=0)

        input_image.append(image)
        input_angle_feats.append(angle_feats)

    input_image = torch.stack(input_image, dim=0)
    # [B, T_img*M=12*M, 1, 3, 224, 224]
    input_image = input_image.unsqueeze(2)
    # image_mask = image_mask.unsqueeze(dim=-1).repeat(
    #     1, 1, 12).view(batch_size,-1).contiguous()

    input_angle_feats = torch.stack(input_angle_feats, dim=0)

    return input_image, image_mask, input_angle_feats





if __name__ == '__main__':
    # soon_data = load_soon_data(anno_file='/mnt/lustre/huangshijia.p/MM/vl_nav/data/SOON/annotations/iccv21_new_released/train.json')
    # 26790
    # r2r_data = load_r2r_data(anno_file='/mnt/lustre/huangshijia.p/MM/vl_nav/data/R2R/annotations/R2R_train_enc.json')
    # 14039
    # reverie_data = load_reverie_data(anno_file='/mnt/lustre/huangshijia.p/MM/vl_nav/data/REVERIE/REVERIE_train.json')
    # 10290
    # eqa_data = load_eqa_data(anno_file='/mnt/lustre/huangshijia.p/MM/vl_nav/data/mp3d/mp3d_eqa_dict.json')
 
    import pdb;pdb.set_trace()