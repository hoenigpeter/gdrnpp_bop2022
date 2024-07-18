# encoding: utf-8
"""This file includes necessary params, info."""
import os.path as osp
import mmcv
import numpy as np

# ---------------------------------------------------------------- #
# ROOT PATH INFO
# ---------------------------------------------------------------- #
cur_dir = osp.abspath(osp.dirname(__file__))
root_dir = osp.normpath(osp.join(cur_dir, ".."))
output_dir = osp.join(root_dir, "output")  # directory storing experiment data (result, model checkpoints, etc).

data_root = osp.join(root_dir, "datasets")
bop_root = osp.join(data_root, "BOP_DATASETS/")

# ---------------------------------------------------------------- #
# ITODD (MVTec ITODD) DATASET
# ---------------------------------------------------------------- #
dataset_root = osp.join(bop_root, "keypose_transparent")
test_dir = osp.join(dataset_root, "test")

model_dir = osp.join(dataset_root, "models")
model_eval_dir = osp.join(dataset_root, "models")
vertex_scale = 0.001
# object info
objects = [str(i) for i in range(1, 16)]
id2obj = {i: str(i) for i in range(1, 16)}

obj_num = len(id2obj)
obj2id = {_name: _id for _id, _name in id2obj.items()}

model_paths = [osp.join(model_dir, "obj_{:06d}.ply").format(_id) for _id in id2obj]
texture_paths = None
model_colors = [((i + 1) * 5, (i + 1) * 5, (i + 1) * 5) for i in range(obj_num)]  # for renderer

diameters = (
    np.array(
        [
            59.6997512787606,
            91.65996818275794,
            159.30904700701274,
            177.035583986581,
            100.13425273939804,
            151.6676201492548,
            69.9558495660679,
            113.64702278929319,
            116.73623335908914,
            119.90105539653153,
            142.52432217780844,
            154.5570838009967,
            163.37200685865733,
            168.77582976589204,
            110.87263490193293
        ]
    )
    / 1000.0
)

# Camera info
width = 1280
height = 960
zNear = 0.25
zFar = 6.0
camera_matrix = np.array([[675.6171332465277, 0.0, 632.1181030273438], [0.0, 675.6171332465277, 338.2853693962097], [0, 0, 1]])
zNear = 0.25
zFar = 6.0


def get_models_info():
    """key is str(obj_id)"""
    models_info_path = osp.join(model_dir, "models_info.json")
    assert osp.exists(models_info_path), models_info_path
    models_info = mmcv.load(models_info_path)  # key is str(obj_id)
    return models_info


# ref core/gdrn_modeling/tools/itodd/itodd_1_compute_fps.py
def get_fps_points():
    fps_points_path = osp.join(model_dir, "fps_points.pkl")
    assert osp.exists(fps_points_path), fps_points_path
    fps_dict = mmcv.load(fps_points_path)
    return fps_dict


# ref core/roi_pvnet/tools/itodd/itodd_1_compute_keypoints_3d.py
def get_keypoints_3d():
    keypoints_3d_path = osp.join(model_dir, "keypoints_3d.pkl")
    assert osp.exists(keypoints_3d_path), keypoints_3d_path
    kpts_dict = mmcv.load(keypoints_3d_path)
    return kpts_dict
