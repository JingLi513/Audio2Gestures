from __future__ import absolute_import
from __future__ import print_function

import os
from glob import glob
import argparse
import numpy as np
import h5py
import time

from fbxtools import FbxTools



SMPLX_JOINT_TO_IDX = {
    "Pelvis": 0,
    "L_Hip": 1,
    "R_Hip": 2,
    "Spine1": 3,
    "L_Knee": 4,
    "R_Knee": 5,
    "Spine2": 6,
    "L_Ankle": 7,
    "R_Ankle": 8,
    "Spine3": 9,
    "L_Foot": 10,
    "R_Foot": 11,
    "Neck": 12,
    "L_Collar": 13,
    "R_Collar": 14,
    "Head": 15,
    "L_Shoulder": 16,
    "R_Shoulder": 17,
    "L_Elbow": 18,
    "R_Elbow": 19,
    "L_Wrist": 20,
    "R_Wrist": 21,
    "Jaw": 22,
    "L_Eye": 23,
    "R_Eye": 24,
    "L_Index1": 25,
    "L_Index2": 26,
    "L_Index3": 27,
    "L_Middle1": 28,
    "L_Middle2": 29,
    "L_Middle3": 30,
    "L_Pinky1": 31,
    "L_Pinky2": 32,
    "L_Pinky3": 33,
    "L_Ring1": 34,
    "L_Ring2": 35,
    "L_Ring3": 36,
    "L_Thumb1": 37,
    "L_Thumb2": 38,
    "L_Thumb3": 39,
    "R_Index1": 40,
    "R_Index2": 41,
    "R_Index3": 42,
    "R_Middle1": 43,
    "R_Middle2": 44,
    "R_Middle3": 45,
    "R_Pinky1": 46,
    "R_Pinky2": 47,
    "R_Pinky3": 48,
    "R_Ring1": 49,
    "R_Ring2": 50,
    "R_Ring3": 51,
    "R_Thumb1": 52,
    "R_Thumb2": 53,
    "R_Thumb3": 54,
}



def save_dataset_into_h5(filename, ds_dict):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    ds_name_list = ds_dict.keys()
    with h5py.File(filename, "w") as f:
            for ds_name in ds_name_list:
                f.create_dataset(ds_name, data=ds_dict[ds_name])



def reorder_joints(data_array, joint_names, joint2num):
    # reorder the joints according to joint2num since the SMPL-X body model uses this order
    assert data_array.shape[1] == len(joint_names) == len(joint2num.keys())
    result = np.zeros_like(data_array)
    for idx, n in enumerate(joint_names):
        assert np.all(result[:, joint2num[n], ...] == 0.0)
        result[:, joint2num[n], ...] = data_array[:, idx, ...]
    return result


def fbx2hdf(config):
    print("Processing ...\n\t{}".format(config.input_name))
    tools = FbxTools(config.input_name)
    print(tools.display_animations())
    t0 = time.time()
    print("Running evaluate_animation_transforms ...")
    animations = tools.evaluate_animation_transforms()
    print("    Elapsed time: {:.3f} sec".format(time.time() - t0))

    data_dict = dict()
    data_dict["framerate"] = tools.frameRate

    names, pos, trans_mats, global_translation = tools.extract_data(animations)

    trans_mats = reorder_joints(trans_mats, names, SMPLX_JOINT_TO_IDX)

    assert np.array_equal(global_translation, pos[:, 0, :])

    data_dict["global_translation"] = global_translation
    print("shape of global_translation: {}".format(data_dict["global_translation"].shape))

    data_dict["LclRotation"] = trans_mats
    print("shape of joint_trans_mats: {}".format(data_dict["LclRotation"].shape))
    print("Frame rate: {}".format(data_dict["framerate"]))
    print("Saving data ...")
    t0 = time.time()
    save_dataset_into_h5(config.output_name, data_dict)
    print("    Elapsed time: {:.3f} sec".format(time.time() - t0))


def parse_cmd_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basepath", "-p", type=str, required=False, help="the path of the input file.")
    return parser.parse_args()


if __name__ == "__main__":
    config = parse_cmd_line_args()
    file_paths = glob(os.path.join(config.basepath,"*.fbx"))

    for file_path in file_paths:
        config.input_name=file_path
        config.output_name = file_path.replace(".fbx",".h5")
        if os.path.exists(config.output_name):
            continue
        fbx2hdf(config)
