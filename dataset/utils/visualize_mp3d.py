import sys
import math
import cv2
import json
from tqdm import tqdm
import numpy as np

def _loc_distance(loc):
    return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)

def mp3d_view(item):
    sys.path.append('/home/zlin/vln/turning/Matterport3DSimulator/build')
    import MatterSim
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60
    VIEWPOINT_SIZE = 12
    connectivity_dir = '/home/zlin/vln/turning/VLN-HAMT/preprocess/connectivity'
    scan_dir = '/media/zlin/2CD830B2D8307C60/Dataset/mp3d'

    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.setDepthEnabled(False)
    sim.initialize()

    scan_id = item['scan']
    path = item['path']
    print(item['navigable_pathViewIds'])
    for vp_id, viewpoint_id in enumerate(path):
        next_vp_id = item['navigable_pathViewIds'][vp_id]

        images = []
        adj_dict = {}
        for ix in range(VIEWPOINT_SIZE):
            if ix == 0:
                sim.newEpisode([scan_id], [viewpoint_id], [0], [math.radians(0)])
            else:
                sim.makeAction([0], [1.0], [0])  # sim.makeAction([location], [heading=30°], [elevation])
            state = sim.getState()[0]
            assert state.viewIndex == ix + 12

            if ix == 0:
                # add current viewpointId
                adj_dict[viewpoint_id] = {
                    'heading': state.heading,
                    'elevation': state.elevation,
                    "normalized_heading": state.heading,
                    "normalized_elevation": state.elevation,
                    'scanId': scan_id,  # sets which scene is used, e.g. "2t7WUuJeko7"
                    'viewpointId': viewpoint_id, # current viewpoint;
                    'pointId': None,
                    'distance': None,
                    'idx': None,
                    'position': (state.location.x, state.location.y, state.location.z),
                }

            image = np.copy(np.array(state.rgb))  # in BGR channel
            images.append(image)

            locations = state.navigableLocations
            for j, loc in enumerate(locations[1:]):
                distance = _loc_distance(loc)
                # Heading and elevation for the loc
                loc_heading = state.heading + loc.rel_heading
                loc_elevation = state.elevation + loc.rel_elevation

                if (loc.viewpointId not in adj_dict or
                        distance < adj_dict[loc.viewpointId]['distance']):
                    adj_dict[loc.viewpointId] = {
                        'heading': loc_heading,
                        'elevation': loc_elevation,
                        "normalized_heading": state.heading + loc.rel_heading,
                        "normalized_elevation": state.elevation + loc.rel_elevation,
                        'scanId': scan_id, # sets which scene is used, e.g. "2t7WUuJeko7"
                        'viewpointId': loc.viewpointId,  # sets the adjacent viewpoint location, e.g. "cc34e9176bfe47ebb23c58c165203134"
                        'pointId': ix, # 当前viewpoint的第ix-th个view指向loc.viewpointId [0-11]
                        'distance': distance,
                        'idx': j + 1, # adjacent index
                        'position': (loc.x, loc.y, loc.z),
                    }

        vrgb = []
        for i in range(len(images)):
            # [480,640,3]
            st_idx = 208  # int(WIDTH/4)
            ed_idx = 640 - st_idx  # int(WIDTH*3/4)
            curimg = images[i][:, st_idx:ed_idx, :]
            id_img = add_id_on_img(curimg, str(i))
            id_img[:, -1, :] = 0

            if i == next_vp_id:
                id_img = add_token_on_img(id_img,token="Next")
            if next_vp_id == -1 and i == 5:
                id_img = add_token_on_img(id_img, token="STOP")
            vrgb.append(id_img)

        # navigator: viewpoint idx [start from 1]
        for k, (vp, vp_dict) in enumerate(adj_dict.items()):
            if vp_dict['pointId'] is None:
                continue
            vrgb[vp_dict['pointId']] = add_vp_on_img(
                vrgb[vp_dict['pointId']], str(vp_dict['pointId'])
            )

        vrgb = vrgb[6:] + vrgb[:6]
        vrgb = np.concatenate(vrgb, axis=1).astype(np.uint8)
        h, w = int(vrgb.shape[0] / 2), int(vrgb.shape[1] / 2)
        vrgb = cv2.resize(vrgb, (w, h), interpolation=cv2.INTER_CUBIC)

        cv2.imshow('RGB', vrgb)
        cv2.waitKey(0)


def add_vp_on_img(img, vp_id):
    img_height = img.shape[0]
    img_width = img.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 2.0
    thickness = 4
    text_width = cv2.getTextSize(str(vp_id), font, font_size, thickness)[0][0]
    start_width = int(img_width / 2 - text_width / 2)
    cv2.putText(
        img,
        str(vp_id),
        (start_width, int(img_height * 3 / 4)),
        font,
        font_size,
        (30, 0, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return img

def add_token_on_img(img,token,color=(255,0,0),height=None):
    img_height = img.shape[0]
    img_width = img.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1.0
    thickness = 2
    text_width = cv2.getTextSize(token, font, font_size, thickness)[0][0]
    start_width = int(img_width / 2 - text_width / 2)
    if height is None:
        height = int(img_height / 5)
    cv2.putText(
        img,
        token,
        # (start_width, int(img_height / 5)),
        (start_width, height),
        font,
        font_size,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )
    return img

def add_id_on_img(img, txt_id):
    img_height = img.shape[0]
    img_width = img.shape[1]
    white = np.ones((10, img.shape[1], 3)) * 255
    img = np.concatenate((img, white), axis=0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1
    thickness = 2
    text_width = cv2.getTextSize(txt_id, font, font_size, thickness)[0][0]
    start_width = int(img_width / 2 - text_width / 2)
    cv2.putText(
        img,
        txt_id,
        (start_width, img_height),
        font,
        font_size,
        (0, 0, 0),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return img