import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import os.path as osp
import sys

import mmcv
import numpy as np
from tqdm import tqdm

cur_dir = osp.abspath(osp.dirname(__file__))
PROJ_ROOT = osp.join(cur_dir, "../../../..")
sys.path.insert(0, PROJ_ROOT)
from lib.egl_renderer.egl_renderer_v3 import EGLRenderer
from lib.vis_utils.image import grid_show
from lib.pysixd import misc
from lib.utils.mask_utils import cocosegm2mask


idx2class = {
    1: "ape",
    2: "benchvise",
    3: "bowl",
    4: "camera",
    5: "can",
    6: "cat",
    7: "cup",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}

class2idx = {_name: _id for _id, _name in idx2class.items()}

classes = idx2class.values()
classes = sorted(classes)

# DEPTH_FACTOR = 1000.
IM_H = 480
IM_W = 640
near = 0.01
far = 6.5

data_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lm/train_pbr"))

cls_indexes = [_idx for _idx in sorted(idx2class.keys())]
cls_names = [idx2class[cls_idx] for cls_idx in cls_indexes]
lm_model_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lm/models"))
model_paths = [osp.join(lm_model_dir, f"obj_{cls_idx:06d}.ply") for cls_idx in cls_indexes]
texture_paths = None

xyz_root = osp.normpath(osp.join(data_dir, "xyz_crop"))
gt_path = osp.join(data_dir, "000000/scene_gt.json")
print(gt_path)
assert osp.exists(gt_path)

K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])


def normalize_to_01(img):
    if img.max() != img.min():
        return (img - img.min()) / (img.max() - img.min())
    else:
        return img


def get_emb_show(bbox_emb):
    show_emb = bbox_emb.copy()
    show_emb = normalize_to_01(bbox_emb)
    return show_emb


class XyzGen(object):
    def __init__(self):
        self.renderer = None

    def get_renderer(self):
        if self.renderer is None:
            self.renderer = EGLRenderer(
                model_paths,
                texture_paths=texture_paths,
                vertex_scale=0.001,
                height=IM_H,
                width=IM_W,
                znear=near,
                zfar=far,
                use_cache=True,
                gpu_id=int(args.gpu),
            )
            self.image_tensor = torch.cuda.FloatTensor(height, width, 4, device=device).detach()
            self.seg_tensor = torch.cuda.FloatTensor(height, width, 4, device=device).detach()
            self.pc_obj_tensor = torch.cuda.FloatTensor(height, width, 4, device=device).detach()
            self.pc_cam_tensor = torch.cuda.FloatTensor(height, width, 4, device=device).detach()
        return self.renderer

    def main(self):
        gt_dict = mmcv.load(gt_path)
        for str_im_id, annos in tqdm(gt_dict.items()):
            int_im_id = int(str_im_id)
            im_path = osp.join(data_dir, f"rgb/{int_im_id:06d}.jpg")

            for anno_i, anno in enumerate(annos):
                obj_id = anno["obj_id"]
                # read Pose
                #pose = np.array(anno["pose"])
                save_path = osp.join(xyz_root, f"{int_im_id:06d}_{anno_i:06d}-xyz.pkl")
                # if osp.exists(save_path) and osp.getsize(save_path) > 0:
                #     continue
                R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                t = np.array(anno["cam_t_m2c"], dtype="float32") / 1000.0
                pose = np.hstack([R, t.reshape(3, 1)])

                K_th = torch.tensor(K, dtype=torch.float32, device=device)
                R_th = torch.tensor(R, dtype=torch.float32, device=device)
                t_th = torch.tensor(t, dtype=torch.float32, device=device)

                cls_name = idx2class[obj_id]
                render_obj_id = cls_indexes.index(obj_id)
                self.get_renderer().render(
                    [render_obj_id],
                    pose,
                    K=K,
                    image_tensor=self.image_tensor,
                    seg_tensor=self.seg_tensor,
                    # pc_obj_tensor=self.pc_obj_tensor,
                    pc_cam_tensor=self.pc_cam_tensor,
                )

                if VIS:
                    bgr_gl = (self.image_tensor[:, :, :3].cpu().numpy() + 0.5).astype(np.uint8)

                mask = (self.seg_tensor[:, :, 0] > 0).to(torch.uint8)
                if mask.sum() == 0:  # NOTE: this should be ignored at training phase
                    imName = osp.basename(im_path)
                    print(f"not visible, cls {cls_name}, im {imName} obj {idx2class[obj_id]} {obj_id}")
                    xyz_info = {
                        "xyz_crop": np.zeros((IM_H, IM_W, 3), dtype=np.float16),
                        "xyxy": [0, 0, IM_W - 1, IM_H - 1],
                    }
                    if VIS:
                        im = mmcv.imread(im_path)

                        mask_gt_rle = anno["mask_full"]
                        mask_gt = cocosegm2mask(mask_gt_rle, IM_H, IM_W)
                        mask_visib_rle = anno["mask_visib"]
                        mask_visib_gt = cocosegm2mask(mask_visib_rle, IM_H, IM_W)

                        show_ims = [
                            bgr_gl[:, :, [2, 1, 0]],
                            im[:, :, [2, 1, 0]],
                            mask_gt,
                            mask_visib_gt,
                        ]
                        show_titles = [
                            "bgr_gl",
                            "im",
                            "mask_gt",
                            "mask_visib_gt",
                        ]
                        grid_show(show_ims, show_titles, row=2, col=2)
                        raise RuntimeError("{}".format(im_path))
                else:
                    ys_xs = mask.nonzero(as_tuple=False)
                    ys, xs = ys_xs[:, 0], ys_xs[:, 1]
                    x1, y1 = [xs.min().item(), ys.min().item()]
                    x2, y2 = [xs.max().item(), ys.max().item()]

                    # xyz_th = self.pc_obj_tensor[:, :, :3].detach()
                    depth_th = self.pc_cam_tensor[:, :, 2].detach()
                    xyz_th = misc.calc_xyz_bp_torch(depth_th, R_th, t_th, K_th)
                    xyz_crop = xyz_th[y1 : y2 + 1, x1 : x2 + 1].cpu().numpy()
                    xyz_info = {
                        "xyz_crop": xyz_crop.astype("float16"),
                        "xyxy": [x1, y1, x2, y2],
                    }

                    if VIS:
                        xyz_th_cpu = xyz_th.cpu().numpy()
                        print(f"xyz min {xyz_th_cpu.min()} max {xyz_th_cpu.max()}")
                        show_ims = [
                            bgr_gl[:, :, [2, 1, 0]],
                            get_emb_show(xyz_th_cpu),
                            get_emb_show(xyz_crop),
                        ]
                        show_titles = ["bgr_gl", "xyz", "xyz_crop"]
                        grid_show(show_ims, show_titles, row=1, col=3)

                mmcv.mkdir_or_exist(osp.dirname(save_path))
                mmcv.dump(xyz_info, save_path)
        if self.renderer is not None:
            self.renderer.close()


if __name__ == "__main__":
    import argparse
    import time

    import setproctitle
    import torch

    parser = argparse.ArgumentParser(description="gen lm egl xyz")
    parser.add_argument("--gpu", type=str, default="0", help="gpu")
    parser.add_argument("--vis", default=False, action="store_true", help="vis")
    args = parser.parse_args()

    height = IM_H
    width = IM_W

    VIS = args.vis

    device = torch.device(int(args.gpu))
    dtype = torch.float32
    tensor_kwargs = {"device": device, "dtype": dtype}

    T_begin = time.perf_counter()
    setproctitle.setproctitle("gen_xyz_lm_egl")
    xyz_gen = XyzGen()
    xyz_gen.main()
    T_end = time.perf_counter() - T_begin
    print("total time: ", T_end)
