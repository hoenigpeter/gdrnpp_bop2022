import os
import os.path as osp
import sys
cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)

import torch
import numpy as np
import cv2
import json

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from core.gdrn_modeling.main_gdrn import Lite
from core.gdrn_modeling.engine.engine_utils import get_out_mask, get_out_coor, batch_data_inference_roi
from core.gdrn_modeling.engine.gdrn_evaluator import get_pnp_ransac_pose
from core.utils.my_checkpoint import MyCheckpointer
from core.utils.data_utils import crop_resize_by_warp_affine, get_2d_coord_np
from lib.vis_utils.image import vis_image_mask_bbox_cv2, vis_image_bboxes_cv2, vis_image_mask_cv2
from lib.utils.utils import iprint
from lib.utils.time_utils import get_time_str
from lib.utils.config_utils import try_get_key
from lib.pysixd import inout, misc
from lib.render_vispy.model3d import load_models

from detectron2.structures import BoxMode
from detectron2.evaluation import inference_context
from detectron2.layers import paste_masks_in_image
from torch.cuda.amp import autocast

from types import SimpleNamespace
from setproctitle import setproctitle
from mmcv import Config
from scipy.spatial.transform import Rotation

import rospy
from core.gdrn_modeling.models import (
    GDRN,
    GDRN_no_region,
    GDRN_cls,
    GDRN_cls2reg,
    GDRN_double_mask,
    GDRN_Dstream_double_mask,
)  # noqa

class GdrnPredictor():
    def __init__(self,
                 config_file_path=osp.join(PROJ_ROOT,"configs/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo.py"),
                 ckpt_file_path=osp.join(PROJ_ROOT,"output/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo/model_final.pth"),
                 #camera_json_path=osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/lmo/camera.json"),
                 camera_intrinsics=[],
                 path_to_obj_models=osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/lmo/models")
                 ):

        self.args = SimpleNamespace(config_file=config_file_path,
                                    opts={'TEST.SAVE_RESULT_ONLY': True,
                                          'MODEL.WEIGHTS': ckpt_file_path},
                                    TEST ={
                                        'TEST.EVAL_PERIOD': 0,
                                        'TEST.VIS': False,
                                        'TEST.USE_PNP': True,
                                        'TEST.USE_DEPTH_REFINE': True,
                                        'TEST.USE_COOR_Z_REFINE': True,
                                        'TEST.TEST_BBOX_TYPE': "est" # gt | est
                                    },
                                    eval_only=True,
                                    fuse=True,
                                    fp16=False,
                                    resume=True,
                                    vertex_scale=0.001,
                                    num_gpus=1,
                                    )


        self.cfg = self.setup(self.args)
        self.objs_dir = path_to_obj_models

        #set your trained object names
        self.objs = {
            1: "002_master_chef_can",  # [1.3360, -0.5000, 3.5105]
            2: "003_cracker_box",  # [0.5575, 1.7005, 4.8050]
            3: "004_sugar_box",  # [-0.9520, 1.4670, 4.3645]
            4: "005_tomato_soup_can",  # [-0.0240, -1.5270, 8.4035]
            5: "006_mustard_bottle",  # [1.2995, 2.4870, -11.8290]
            6: "007_tuna_fish_can",  # [-0.1565, 0.1150, 4.2625]
            7: "008_pudding_box",  # [1.1645, -4.2015, 3.1190]
            8: "009_gelatin_box",  # [1.4460, -0.5915, 3.6085]
            9: "010_potted_meat_can",  # [2.4195, 0.3075, 8.0715]
            10: "011_banana",  # [-18.6730, 12.1915, -1.4635]
            11: "019_pitcher_base",  # [5.3370, 5.8855, 25.6115]
            12: "021_bleach_cleanser",  # [4.9290, -2.4800, -13.2920]
            13: "024_bowl",  # [-0.2270, 0.7950, -2.9675]
            14: "025_mug",  # [-8.4675, -0.6995, -1.6145]
            15: "035_power_drill",  # [9.0710, 20.9360, -2.1190]
            16: "036_wood_block",  # [1.4265, -2.5305, 17.1890]
            17: "037_scissors",  # [7.0535, -28.1320, 0.0420]
            18: "040_large_marker",  # [0.0460, -2.1040, 0.3500]
            19: "051_large_clamp",  # [10.5180, -1.9640, -0.4745]
            20: "052_extra_large_clamp",  # [-0.3950, -10.4130, 0.1620]
            21: "061_foam_brick",  # [-0.0805, 0.0805, -8.2435]
        }

        self.cls_names = [i for i in self.objs.values()]
        self.obj_ids = [i for i in self.objs.keys()]
        self.extents = self._get_extents()

        self.cam = camera_intrinsics
        self.depth_scale = 0.1

        model_lite = Lite(
            accelerator="gpu",
            strategy=None,
            devices=1,
            num_nodes=1,
            precision=32)

        self.model = self.set_eval_model(model_lite, self.args, self.cfg)

        # if self.cfg.TEST.USE_DEPTH_REFINE:
        #     self.ren = Renderer(size=(64, 64), cam=self.cam)
        #     print([os.path.join(self.objs_dir, f"obj_{i:06d}.ply") for i in self.objs.keys()])
        #     self.ren_models = load_models(
        #         model_paths=[os.path.join(self.objs_dir, f"obj_{i:06d}.ply") for i in self.objs.keys()],
        #         scale_to_meter=0.001,
        #         cache_dir=".cache",
        #         #texture_paths=None,
        #         texture_paths=self.data_ref.texture_paths if self.cfg.DEBUG else None,
        #         center=False,
        #         use_cache=True,
        #     )

        self.ren_models = load_models(
            model_paths=[os.path.join(self.objs_dir, f"obj_{i:06d}.ply") for i in self.objs.keys()],
            scale_to_meter=0.001,
            cache_dir=".cache",
            #texture_paths=None,
            texture_paths=self.data_ref.texture_paths if self.cfg.DEBUG else None,
            center=False,
            use_cache=True,
        )

    def set_eval_model(self, model_lite, args, cfg):
        model_lite.set_my_env(args, cfg)

        model, optimizer = eval(cfg.MODEL.POSE_NET.NAME).build_model_optimizer(cfg, is_test=args.eval_only)

        MyCheckpointer(model, save_dir=cfg.OUTPUT_DIR, prefix_to_remove="_module.").resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )

        return model

    def inference(self, data_dict):
        """
        Run gdrn model inference.
        Args:
            data_dict: input of the model

        Returns:
            dict: output of the model
        """
        with inference_context(self.model), torch.no_grad():
            with autocast(enabled=False):  # gdrn amp_test seems slower
                out_dict = self.model(
                    data_dict["roi_img"],
                    roi_classes=data_dict["roi_cls"],
                    roi_cams=data_dict["roi_cam"],
                    roi_whs=data_dict["roi_wh"],
                    roi_centers=data_dict["roi_center"],
                    resize_ratios=data_dict["resize_ratio"],
                    roi_coord_2d=data_dict.get("roi_coord_2d", None),
                    roi_coord_2d_rel=data_dict.get("roi_coord_2d_rel", None),
                    roi_extents=data_dict.get("roi_extent", None),
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        return out_dict

    def postprocessing(self, data_dict, out_dict, renderer_request_queue, renderer_result_queue):
        """
        Postprocess the gdrn model outputs
        Args:
            data_dict: gdrn model preprocessed data
            out_dict: gdrn model output
        Returns:
            dict: poses of objects
        """
        i_out = -1
        data_dict["cur_res"] = []
        for i_inst in range(len(data_dict["roi_img"])):
            i_out += 1

            cur_obj_id = self.obj_ids[int(data_dict["roi_cls"][i_out])]
            cur_res = {
                "obj_id": cur_obj_id,
                "score": float(data_dict["score"][i_inst]),
                "bbox_est": data_dict["bbox_est"][i_inst].detach().cpu().numpy(),  # xyxy
            }
            if self.cfg.TEST.USE_PNP:
                pose_est_pnp = get_pnp_ransac_pose(self.cfg, data_dict, out_dict, i_inst, i_out)
                cur_res["R"] = pose_est_pnp[:3, :3]
                cur_res["t"] = pose_est_pnp[:3, 3]
            else:
                cur_res.update(
                    {
                        "R": out_dict["rot"][i_out].detach().cpu().numpy(),
                        "t": out_dict["trans"][i_out].detach().cpu().numpy(),
                    }
                )
            data_dict["cur_res"].append(cur_res)

        if self.cfg.TEST.USE_DEPTH_REFINE:
           self.process_depth_refine(data_dict, out_dict, renderer_request_queue, renderer_result_queue)

        poses = {}
        for res in data_dict["cur_res"]:
            pose = np.eye(4)
            pose[:3, :3] = res['R']
            pose[:3, 3] = res['t']
            poses[self.objs.get(res['obj_id'])] = pose

        return poses

    def process_depth_refine(self, inputs, out_dict, renderer_request_queue, renderer_result_queue):
        """
        Postprocess the gdrn result and refine with the depth information.
        Args:
            inputs: gdrn model input
            out_dict: gdrn model output
        """
        cfg = self.cfg
        net_cfg = cfg.MODEL.POSE_NET
        out_coor_x = out_dict["coor_x"].detach()
        out_coor_y = out_dict["coor_y"].detach()
        out_coor_z = out_dict["coor_z"].detach()
        out_xyz = get_out_coor(cfg, out_coor_x, out_coor_y, out_coor_z)
        out_xyz = out_xyz.to(torch.device("cpu")) #.numpy()

        out_mask = get_out_mask(cfg, out_dict["mask"].detach())
        out_mask = out_mask.to(torch.device("cpu")) #.numpy()
        out_rots = out_dict["rot"].detach().to(torch.device("cpu")).numpy()
        out_transes = out_dict["trans"].detach().to(torch.device("cpu")).numpy()

        zoom_K = batch_data_inference_roi(cfg, [inputs])['roi_zoom_K']

        out_i = -1
      
        pub_sensor = rospy.Publisher('/depth_sensor_mask', Image, queue_size=10)
        pub_render = rospy.Publisher('/depth_render_mask', Image, queue_size=10)
        bridge = CvBridge()
        
        for i, _input in enumerate([inputs]):

            for inst_i in range(len(_input["roi_img"])):
                out_i += 1

                K_crop = zoom_K[inst_i].cpu().numpy().copy()
                # print('K_crop', K_crop)

                roi_label = _input["roi_cls"][inst_i]  # 0-based label
                roi_label, cls_name = self._maybe_adapt_label_cls_name(roi_label)
                if cls_name is None:
                    continue

                # get pose
                xyz_i = out_xyz[out_i].permute(1, 2, 0)
                mask_i = np.squeeze(out_mask[out_i])

                rot_est = out_rots[out_i]
                trans_est = out_transes[out_i]
                pose_est = np.hstack([rot_est, trans_est.reshape(3, 1)])
                
                rot_est = inputs["cur_res"][inst_i]["R"]
                trans_est = inputs["cur_res"][inst_i]["t"]
                pose_est = np.hstack([rot_est, trans_est.reshape(3,1)])
                #depth_sensor_crop = _input['roi_depth'][inst_i].cpu().numpy().copy().squeeze()
                depth_sensor_crop = cv2.resize(_input['roi_depth'][inst_i].cpu().numpy().copy().squeeze(), (64, 64))
                depth_sensor_mask_crop = depth_sensor_crop > 0

                #TODO find out why depth_sensor_corp does not have the correct scale
                depth_sensor_crop = depth_sensor_crop * 10

                depth_sensor_mask_crop_image = (depth_sensor_mask_crop * 255).astype(np.uint8)
                depth_sensor_crop_ros = bridge.cv2_to_imgmsg(depth_sensor_mask_crop_image, encoding='passthrough')
                pub_sensor.publish(depth_sensor_crop_ros)

                net_cfg = cfg.MODEL.POSE_NET
                crop_res = net_cfg.OUTPUT_RES
                for j in range(cfg.TEST.DEPTH_REFINE_ITER):
                    renderer_request_queue.put([
                        K_crop,
                        self.ren_models[self.cls_names.index(cls_name)],
                        pose_est
                    ])
                    
                    while renderer_result_queue.empty():
                        rospy.sleep(0.03)
                    ren_dp = renderer_result_queue.get(block=True, timeout=0.2)
                    
                    ren_mask = ren_dp > 0

                    if self.cfg.TEST.USE_COOR_Z_REFINE:
                        coor_np = xyz_i.numpy()
                        coor_np_t = coor_np.reshape(-1, 3)
                        coor_np_t = coor_np_t.T
                        coor_np_r = rot_est @ coor_np_t
                        coor_np_r = coor_np_r.T
                        coor_np_r = coor_np_r.reshape(crop_res, crop_res, 3)
                        query_img_norm = coor_np_r[:, :, -1] * mask_i.numpy()
                        query_img_norm = query_img_norm * ren_mask * depth_sensor_mask_crop
                    else:
                        query_img = xyz_i

                        query_img_norm = torch.norm(query_img, dim=-1) * mask_i
                        query_img_norm = query_img_norm.numpy() * ren_mask * depth_sensor_mask_crop
                    norm_sum = query_img_norm.sum()
                    if norm_sum == 0:
                        continue
                    query_img_norm /= norm_sum
                    norm_mask = query_img_norm > (query_img_norm.max() * 0.8)
                    yy, xx = np.argwhere(norm_mask).T  # 2 x (N,)

                    depth_diff = depth_sensor_crop[yy, xx] - ren_dp[yy, xx]
                    depth_adjustment = np.median(depth_diff)

                    yx_coords = np.meshgrid(np.arange(crop_res), np.arange(crop_res))
                    yx_coords = np.stack(yx_coords[::-1], axis=-1)  # (crop_res, crop_res, 2yx)
                    yx_ray_2d = (yx_coords * query_img_norm[..., None]).sum(axis=(0, 1))  # y, x
                    ray_3d = np.linalg.inv(K_crop) @ (*yx_ray_2d[::-1], 1)
                    ray_3d /= ray_3d[2]

                    trans_delta = ray_3d[:, None] * depth_adjustment
                    trans_est = trans_est + trans_delta.reshape(3)
                    pose_est = np.hstack([rot_est, trans_est.reshape(3, 1)])
                inputs["cur_res"][inst_i]["R"] = pose_est[:3,:3]
                inputs["cur_res"][inst_i]["t"] = pose_est[:3,3]
                self.publish_mesh_marker(cls_name, pose_est[:3, :3], pose_est[:3, 3])
        
    def publish_mesh_marker(self, cls_name, R_est, t_est):
        from visualization_msgs.msg import Marker
        vis_pub = rospy.Publisher("/gdrnet_meshes", Marker, latch=True)
        model = self.ren_models[self.cls_names.index(cls_name)]
        model_vertices = np.array(model.vertices)
        model_colors = model.colors
        marker = Marker()
        marker.header.frame_id = 'head_rgbd_sensor_rgb_frame'
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.TRIANGLE_LIST
        marker.ns = cls_name
        marker.action = Marker.ADD
        marker.pose.position.x = t_est[0]
        marker.pose.position.y = t_est[1]
        marker.pose.position.z = t_est[2]
        quat = Rotation.from_matrix(R_est).as_quat()
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA
        assert model_vertices.shape[0] == model_colors.shape[0]

        # TRIANGLE_LIST needs 3*x points to render x triangles 
        # => find biggest number smaller than model_vertices.shape[0] that is still divisible by 3
        shape_vertices = 3*int((model_vertices.shape[0] - 1)/3)
        for i in range(shape_vertices):
            pt = Point(x = model_vertices[i, 0], y = model_vertices[i, 1], z = model_vertices[i, 2])
            marker.points.append(pt)
            rgb = ColorRGBA(r = 0, g = 0, b = 1, a = 1.0)
            marker.colors.append(rgb)

        vis_pub.publish(marker)
        


    def normalize_image(self, cfg, image):
        """
        cfg: upper format, the whole cfg; lower format, the input_cfg
        image: CHW format
        """
        pixel_mean = np.array(try_get_key(cfg, "MODEL.PIXEL_MEAN", "pixel_mean")).reshape(-1, 1, 1)
        pixel_std = np.array(try_get_key(cfg, "MODEL.PIXEL_STD", "pixel_std")).reshape(-1, 1, 1)
        return (image - pixel_mean) / pixel_std

    def _maybe_adapt_label_cls_name(self, label):
        cls_name = self.cls_names[label]
        return label, cls_name

    def preprocessing(self, outputs, image, depth_img):
        """
        Preprocessing detection model output and input image
        Args:
            outputs: yolo model output
            image: rgb image
            depth_img: depth image
        Returns:
            dict
        """

        dataset_dict = {
            "cam": self.cam,
            "annotations": []
        }

        if outputs is None:
            # TODO set default output
            return None

        if len(outputs[0].size()) == 1:
            #outputs[0].unsqueeze(0)
            boxes = outputs[0].unsqueeze(0).cpu()
        else:
            boxes = outputs[0].cpu()
        
        for i in range(boxes.size()[0]):
            annot_inst = {}
            box = boxes[i].tolist()
            annot_inst["category_id"] = int(box[6])
            annot_inst["score"] = box[4] * box[5]
            annot_inst["bbox_est"] = [box[0], box[1], box[2], box[3]]
            annot_inst["bbox_mode"] = BoxMode.XYXY_ABS

            dataset_dict["annotations"].append(annot_inst)

        im_H_ori, im_W_ori = image.shape[:2]

        # other transforms (mainly geometric ones);
        # for 6d pose task, flip is not allowed in general except for some 2d keypoints methods
        im_H, im_W = image.shape[:2]  # h, w

        # NOTE: scale camera intrinsic if necessary ================================
        scale_x = im_W / im_W_ori
        scale_y = im_H / im_H_ori  # NOTE: generally scale_x should be equal to scale_y
        if "cam" in dataset_dict:
            if im_W != im_W_ori or im_H != im_H_ori:
                dataset_dict["cam"][0] *= scale_x
                dataset_dict["cam"][1] *= scale_y
            K = dataset_dict["cam"].astype("float32")
            dataset_dict["cam"] = torch.as_tensor(K)
        else:
            raise RuntimeError("cam intrinsic is missing")

        if depth_img is not None:
            depth_img = (depth_img / (1 / self.depth_scale))
            depth_ch = 1
            depth = depth_img.reshape(im_H, im_W, depth_ch).astype("float32")
        input_res = 256
        out_res = 64

        # CHW -> HWC
        coord_2d = get_2d_coord_np(im_W, im_H, low=0, high=1).transpose(1, 2, 0)
        #################################################################################
        # here get batched rois
        roi_infos = {}
        # yapf: disable
        roi_keys = ["scene_im_id", "file_name", "cam", "im_H", "im_W",
                    "roi_img", "inst_id", "roi_coord_2d", "roi_coord_2d_rel",
                    "roi_cls", "score", "time", "roi_extent",
                    "bbox_est", "bbox_mode", "bbox_center", "roi_wh",
                    "scale", "resize_ratio", "model_info",
                    ]

        if depth_img is not None:
            roi_keys.append("roi_depth")
        for _key in roi_keys:
            roi_infos[_key] = []

        #for inst_i, inst_infos in enumerate(dataset_dict["annotations"]):
        #    print(inst_infos["category_id"])

        for inst_i, inst_infos in enumerate(dataset_dict["annotations"]):
            # inherent image-level infos
            roi_infos["im_H"].append(im_H)
            roi_infos["im_W"].append(im_W)
            roi_infos["cam"].append(dataset_dict["cam"].cpu().numpy())

            # roi-level infos
            roi_infos["inst_id"].append(inst_i)
            # roi_infos["model_info"].append(inst_infos["model_info"])

            roi_cls = inst_infos["category_id"]
            roi_infos["roi_cls"].append(roi_cls)
            roi_infos["score"].append(inst_infos.get("score", 1.0))

            roi_infos["time"].append(inst_infos.get("time", 0))

            # extent
            roi_extent = self.extents[roi_cls]
            roi_infos["roi_extent"].append(roi_extent)

            # TODO: adjust amodal bbox here
            bbox = np.array(inst_infos["bbox_est"])
            roi_infos["bbox_est"].append(bbox)
            roi_infos["bbox_mode"].append(BoxMode.XYXY_ABS)
            x1, y1, x2, y2 = bbox
            bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
            bw = max(x2 - x1, 1)
            bh = max(y2 - y1, 1)
            scale = max(bh, bw) * 1.5 #cfg.INPUT.DZI_PAD_SCALE
            scale = min(scale, max(im_H, im_W)) * 1.0

            roi_infos["bbox_center"].append(bbox_center.astype("float32"))
            roi_infos["scale"].append(scale)
            roi_wh = np.array([bw, bh], dtype=np.float32)
            roi_infos["roi_wh"].append(roi_wh)
            roi_infos["resize_ratio"].append(out_res / scale)

            # CHW, float32 tensor
            # roi_image
            roi_img = crop_resize_by_warp_affine(
                image, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
            ).transpose(2, 0, 1)

            roi_img = self.normalize_image(self.cfg, roi_img)
            roi_infos["roi_img"].append(roi_img.astype("float32"))

            # roi_depth
            if depth_img is not None:
                roi_depth = crop_resize_by_warp_affine(
                    depth, bbox_center, scale, input_res, interpolation=cv2.INTER_NEAREST
                )
                if depth_ch == 1:
                    roi_depth = roi_depth.reshape(1, input_res, input_res)
                else:
                    roi_depth = roi_depth.transpose(2, 0, 1)
                roi_infos["roi_depth"].append(roi_depth.astype("float32"))

            # roi_coord_2d
            roi_coord_2d = crop_resize_by_warp_affine(
                coord_2d, bbox_center, scale, out_res, interpolation=cv2.INTER_LINEAR
            ).transpose(
                2, 0, 1
            )  # HWC -> CHW
            roi_infos["roi_coord_2d"].append(roi_coord_2d.astype("float32"))

        for _key in roi_keys:
            if _key in ["roi_img", "roi_coord_2d", "roi_coord_2d_rel", "roi_depth"]:
                dataset_dict[_key] = torch.as_tensor(np.array(roi_infos[_key])).contiguous()
            elif _key in ["model_info", "scene_im_id", "file_name"]:
                # can not convert to tensor
                dataset_dict[_key] = roi_infos[_key]
            else:
                if isinstance(roi_infos[_key], list):
                    dataset_dict[_key] = torch.as_tensor(np.array(roi_infos[_key]))
                else:
                    dataset_dict[_key] = torch.as_tensor(roi_infos[_key])

        roi_keys = ["im_H", "im_W",
                    "roi_img", "inst_id", "roi_coord_2d", "roi_coord_2d_rel",
                    "roi_cls", "score", "time", "roi_extent",
                    "bbox", "bbox_est", "bbox_mode", "roi_wh",
                    "scale", "resize_ratio",
                    ]
        batch = {}

        if depth_img is not None:
            roi_keys.append("roi_depth")

        for key in roi_keys:
            if key in ["roi_cls"]:
                dtype = torch.long
            else:
                dtype = torch.float32
            if key in dataset_dict:
                batch[key] = torch.cat([dataset_dict[key]], dim=0).to(device='cuda', dtype=dtype, non_blocking=True)

        batch["roi_cam"] = torch.cat([dataset_dict["cam"]], dim=0).to('cuda', non_blocking=True)
        batch["roi_center"] = torch.cat([dataset_dict["bbox_center"]], dim=0).to('cuda', non_blocking=True)
        batch["bbox_center"] = torch.cat([dataset_dict["bbox_center"]], dim=0).to('cuda', non_blocking=True)#roi_infos["bbox_center"]
        batch["cam"] = dataset_dict["cam"]

        return batch

    def _get_extents(self):
        """label based keys."""
        self.obj_models = {}

        cur_extents = {}
        idx = 0
        for obj_key in self.objs.keys():
            model_path = os.path.join(self.objs_dir, f"obj_{obj_key:06d}.ply")
            model = inout.load_ply(model_path, vertex_scale=self.args.vertex_scale)
            self.obj_models[obj_key] = model
            pts = model["pts"]
            xmin, xmax = np.amin(pts[:, 0]), np.amax(pts[:, 0])
            ymin, ymax = np.amin(pts[:, 1]), np.amax(pts[:, 1])
            zmin, zmax = np.amin(pts[:, 2]), np.amax(pts[:, 2])
            size_x = xmax - xmin
            size_y = ymax - ymin
            size_z = zmax - zmin
            cur_extents[idx] = np.array([size_x, size_y, size_z], dtype="float32")
            idx += 1

        return cur_extents

    def setup(self, args):
        """Create configs and perform basic setups."""
        cfg = Config.fromfile(args.config_file)
        if args.opts is not None:
            cfg.merge_from_dict(args.opts)
        if args.TEST is not None:
            cfg.merge_from_dict(args.TEST)
        ############## pre-process some cfg options ######################
        # NOTE: check if need to set OUTPUT_DIR automatically
        if cfg.OUTPUT_DIR.lower() == "auto":
            cfg.OUTPUT_DIR = os.path.join(
                cfg.OUTPUT_ROOT,
                os.path.splitext(args.config_file)[0].split("configs/")[1],
            )
            iprint(f"OUTPUT_DIR was automatically set to: {cfg.OUTPUT_DIR}")

        if cfg.get("EXP_NAME", "") == "":
            setproctitle("{}.{}".format(os.path.splitext(os.path.basename(args.config_file))[0], get_time_str()))
        else:
            setproctitle("{}.{}".format(cfg.EXP_NAME, get_time_str()))

        if cfg.SOLVER.AMP.ENABLED:
            if torch.cuda.get_device_capability() <= (6, 1):
                iprint("Disable AMP for older GPUs")
                cfg.SOLVER.AMP.ENABLED = False

        # NOTE: pop some unwanted configs in detectron2
        # ---------------------------------------------------------
        cfg.SOLVER.pop("STEPS", None)
        cfg.SOLVER.pop("MAX_ITER", None)
        bs_ref = cfg.SOLVER.get("REFERENCE_BS", cfg.SOLVER.IMS_PER_BATCH)  # nominal batch size
        if bs_ref <= cfg.SOLVER.IMS_PER_BATCH:
            bs_ref = cfg.SOLVER.REFERENCE_BS = cfg.SOLVER.IMS_PER_BATCH
            # default DDP implementation is slow for accumulation according to: https://pytorch.org/docs/stable/notes/ddp.html
            # all-reduce operation is carried out during loss.backward().
            # Thus, there would be redundant all-reduce communications in a accumulation procedure,
            # which means, the result is still right but the training speed gets slower.
            # TODO: If acceleration is needed, there is an implementation of allreduce_post_accumulation
            # in https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/run_pretraining.py
            accumulate_iter = max(round(bs_ref / cfg.SOLVER.IMS_PER_BATCH), 1)  # accumulate loss before optimizing
        else:
            accumulate_iter = 1
        # NOTE: get optimizer from string cfg dict
        if cfg.SOLVER.OPTIMIZER_CFG != "":
            if isinstance(cfg.SOLVER.OPTIMIZER_CFG, str):
                optim_cfg = eval(cfg.SOLVER.OPTIMIZER_CFG)
                cfg.SOLVER.OPTIMIZER_CFG = optim_cfg
            else:
                optim_cfg = cfg.SOLVER.OPTIMIZER_CFG
            iprint("optimizer_cfg:", optim_cfg)
            cfg.SOLVER.OPTIMIZER_NAME = optim_cfg["type"]
            cfg.SOLVER.BASE_LR = optim_cfg["lr"]
            cfg.SOLVER.MOMENTUM = optim_cfg.get("momentum", 0.9)
            cfg.SOLVER.WEIGHT_DECAY = optim_cfg.get("weight_decay", 1e-4)
            if accumulate_iter > 1:
                if "weight_decay" in cfg.SOLVER.OPTIMIZER_CFG:
                    cfg.SOLVER.OPTIMIZER_CFG["weight_decay"] *= (
                            cfg.SOLVER.IMS_PER_BATCH * accumulate_iter / bs_ref
                    )  # scale weight_decay
        if accumulate_iter > 1:
            cfg.SOLVER.WEIGHT_DECAY *= cfg.SOLVER.IMS_PER_BATCH * accumulate_iter / bs_ref
        # -------------------------------------------------------------------------
        if cfg.get("DEBUG", False):
            iprint("DEBUG")
            args.num_gpus = 1
            args.num_machines = 1
            cfg.DATALOADER.NUM_WORKERS = 0
            cfg.TRAIN.PRINT_FREQ = 1

        exp_id = "{}".format(os.path.splitext(os.path.basename(args.config_file))[0])

        if args.eval_only:
            if cfg.TEST.USE_PNP:
                # NOTE: need to keep _test at last
                exp_id += "{}_test".format(cfg.TEST.PNP_TYPE.upper())
            else:
                exp_id += "_test"
        cfg.EXP_ID = exp_id
        cfg.RESUME = args.resume
        ####################################
        # cfg.freeze()
        return cfg

    # def gdrn_visualization(self, batch, out_dict, image):
    #     vis_dict = {}

    #     # for crop and resize
    #     bs = batch["roi_cls"].shape[0]
    #     #print(bs)
    #     tensor_kwargs = {"dtype": torch.float32, "device": "cuda"}
    #     rois_xy0 = batch["roi_center"] - batch["scale"].view(bs, -1) / 2  # bx2
    #     rois_xy1 = batch["roi_center"] + batch["scale"].view(bs, -1) / 2  # bx2
    #     batch["inst_rois"] = torch.cat([torch.arange(bs, **tensor_kwargs).view(-1, 1), rois_xy0, rois_xy1], dim=1)

    #     im_H = int(batch["im_H"][0])
    #     im_W = int(batch["im_W"][0])
    #     if "full_mask" in out_dict:
    #         raw_full_masks = out_dict["full_mask"]
    #         full_mask_probs = get_out_mask(self.cfg, raw_full_masks)
    #         full_masks_in_im = paste_masks_in_image(
    #             full_mask_probs[:, 0, :, :],
    #             batch["inst_rois"][:, 1:5],
    #             image_shape=(im_H, im_W),
    #             threshold=0.5,
    #         )
    #         full_masks_np = full_masks_in_im.detach().to(torch.uint8).cpu().numpy()

    #         img_vis_full_mask = vis_image_mask_bbox_cv2(
    #             image,
    #             [full_masks_np[i] for i in range(bs)],
    #             [batch["bbox_est"][i].detach().cpu().numpy() for i in range(bs)],
    #             labels=self.cls_names,
    #         )

    #         vis_dict[f"im_det_and_mask_full"] = img_vis_full_mask[:, :, ::-1]

    #     for i in range(bs):
    #         R = batch["cur_res"][i]["R"]
    #         t = batch["cur_res"][i]["t"]
    #         # pose_est = np.hstack([R, t.reshape(3, 1)])
    #         proj_pts_est = misc.project_pts(self.obj_models[i+1]["pts"], self.cam, R, t)
    #         mask_pose_est = misc.points2d_to_mask(proj_pts_est, im_H, im_W)
    #         image_mask_pose_est = vis_image_mask_cv2(image, mask_pose_est, color="yellow" if i == 0 else "blue")
    #         image_mask_pose_est = vis_image_bboxes_cv2(
    #             image_mask_pose_est,
    #             [batch["bbox_est"][i].detach().cpu().numpy()],
    #             labels=[self.cls_names[i]]
    #         )
    #         vis_dict[f"im_{i}_mask_pose_est"] = image_mask_pose_est[:, :, ::-1]
    #     show_ims = np.hstack([cv2.cvtColor(_v, cv2.COLOR_BGR2RGB) for _k, _v in vis_dict.items()])
    #     cv2.imshow('result', show_ims)
    #     cv2.waitKey(1)