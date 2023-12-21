"""keyframe.txt in ycb dataset, the sub-test set used in papers."""

import mmcv
import os.path as osp
import json
import yaml

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../../.."))


data_root = "datasets/BOP_DATASETS/ycbv"
ycbv_path = osp.join(PROJ_ROOT, data_root)
object_id_file = osp.join(ycbv_path, 'data/ycbv.yaml')
with open(object_id_file, 'r') as stream:
    data_loaded = yaml.safe_load(stream)
IDX2CLASS = data_loaded['names']
CLASSES = IDX2CLASS.values()
CLASSES = list(sorted(CLASSES))
CLASS2IDX = {cls_name: idx for idx, cls_name in IDX2CLASS.items()}


keyframe_path = osp.join(data_root, "image_sets/keyframe.txt")
# can get it from either the original ycb dataset or
# https://github.com/yuxng/YCB_Video_toolbox/blob/master/keyframe.txt
#   contents: each line looks like 0048/000001
assert osp.exists(keyframe_path), keyframe_path


def main():
    with open(keyframe_path, "r") as f:
        test_keyframe_indices = [line.strip("\r\n") for line in f.readlines()]

    test_targets = []  # {"im_id": , "inst_count": , "obj_id": , "scene_id": }
    num_test_im = 0
    test_scenes = [i for i in range(48, 59 + 1)]
    for scene_id in test_scenes:
        print("scene_id", scene_id)
        BOP_gt_file = osp.join(data_root, f"test/{scene_id:06d}/scene_gt.json")
        assert osp.exists(BOP_gt_file), BOP_gt_file
        gt_dict = mmcv.load(BOP_gt_file)
        all_ids = [int(k) for k in gt_dict.keys()]
        print(len(all_ids))

        for idx in all_ids:
            scene_im_id_str = f"{int(scene_id):04d}/{int(idx):06d}"
            if scene_im_id_str not in test_keyframe_indices:
                continue
            annos = gt_dict[str(idx)]
            obj_ids = [anno["obj_id"] for anno in annos]
            num_inst_dict = {}
            # stat num instances for each obj
            for obj_id in obj_ids:
                if obj_id not in num_inst_dict:
                    num_inst_dict[obj_id] = 1
                else:
                    num_inst_dict[obj_id] += 1
            for obj_id in num_inst_dict:
                target = {
                    "im_id": idx,
                    "inst_count": num_inst_dict[obj_id],
                    "obj_id": obj_id,
                    "scene_id": scene_id,
                }
                test_targets.append(target)
            num_test_im += 1
    res_file = osp.join(cur_dir, "ycbv_test_targets_keyframe.json")
    print(res_file)
    print(len(test_targets))
    print("num test images: ", num_test_im)  # 2949
    # mmcv.dump(test_targets, res_file)
    with open(res_file, "w") as f:
        f.write("[\n" + ",\n".join(json.dumps(item) for item in test_targets) + "]\n")
    print("done")


if __name__ == "__main__":
    main()
