import json


def load_r2r_data(anno_file):
    assert anno_file.exists()
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
            new_item['instr_id'] = '{}_{}'.format(i,j)
            new_item['instruction'] = instr
            del new_item['instructions']
            del new_item['instr_encodings']
            new_data.append(new_item)
            sample_index += 1
    return new_data


def load_reverie_data(anno_file):
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
            del new_item['instr_encodings']
            new_data.append(new_item)
    return new_data


def load_soon_data(anno_file):
    assert anno_file.exists()
    with open(str(anno_file)) as f:
        data = json.load(f)
    new_data = []
    sample_index = 0
    for i, item in enumerate(data):
        for j, path in enumerate(item['path']):
            # Split multiple instructions into separate entries
            for k, instr in enumerate(item['instructions']):
                new_item = dict()
                new_item['sample_idx'] = sample_index
                # soon: idx-path_idx-instr_idx
                new_item['soon_idx'] = "{}_{}_{}".format(i,j,k)
                # soon instructions
                new_item['instruction'] = instr
                # current path
                new_item['path'] = path
                new_data.append(new_item)
    return new_data


def load_fr2r_data(anno_file):
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
            del new_item['instr_encodings']
            new_data.append(new_item)
    return new_data
