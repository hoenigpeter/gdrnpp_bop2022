# inference with detector, gdrn, and refiner
import os.path as osp
import sys
import torch
cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)

from predictor_gdrn import GdrnPredictor
import os
import argparse
import time

import cv2
import numpy as np

import rospy
from std_msgs.msg import Header
from object_detector_msgs.msg import BoundingBox, Detection, Detections, PoseWithConfidence
from object_detector_msgs.srv import detectron2_service_server, estimate_poses, estimate_posesResponse
from geometry_msgs.msg import Pose, Point, Quaternion
import actionlib
from robokudo_msgs.msg import GenericImgProcAnnotatorResult, GenericImgProcAnnotatorAction
import json
from cv_bridge import CvBridge, CvBridgeError
import tf
from lib.render_vispy.renderer import RendererROS
import queue

class GDRN_ROS:
    def __init__(self, renderer_request_queue, renderer_result_queue, dataset_name):
            intrinsics = np.asarray(rospy.get_param('/pose_estimator/intrinsics'))
            self.frame_id = rospy.get_param('/pose_estimator/color_frame_id')
            self.gdrn_predictor = GdrnPredictor(
                config_file_path=osp.join(PROJ_ROOT,"configs/gdrn/" + dataset_name + "/" + dataset_name + "_inference.py"),
                ckpt_file_path=osp.join(PROJ_ROOT,"output/gdrnpp_" + dataset_name + "_weights.pth"),
                camera_intrinsics=intrinsics,
                path_to_obj_models=osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/" + dataset_name + "/models"),
                model_string=dataset_name
            )

            self.renderer_request_queue = renderer_request_queue
            self.renderer_result_queue = renderer_result_queue
            rospy.init_node("gdrn_estimation")
            self.server = actionlib.SimpleActionServer('/pose_estimator/gdrnet', 
                                                        GenericImgProcAnnotatorAction, 
                                                        execute_cb=self.estimate_pose, 
                                                        auto_start=False)
            self.server.start()
            print("Pose Estimation with GDRNPP is ready.")
    
    """
    When using the robokudo_msgs, as the callback function for the action server
    """
    def estimate_pose(self, req):
        #print("request detection...")
        start_time = time.time()

        # === IN ===
        # --- rgb
        bb_detections = req.bb_detections
        class_names = req.class_names
        description = req.description
        scores = json.loads(description)
        rgb = req.rgb
        depth = req.depth

        width, height = rgb.width, rgb.height
        assert width == 640 and height == 480

        try:
            image = CvBridge().imgmsg_to_cv2(rgb, "bgr8")
        except CvBridgeError as e:
            print(e)

        try:
            depth.encoding = "mono16"
            depth_img = CvBridge().imgmsg_to_cv2(depth, "mono16")
            depth_img = depth_img/1000
        except CvBridgeError as e:
            print(e)

        valid_class_names = []
        pose_results = []
        class_confidences = []
        for name, roi in zip(class_names, bb_detections):
            if name == "036_wood_block":
                continue
            score = np.float32(scores[name])

            obj_id = -1
            for number in self.gdrn_predictor.objs:
                if self.gdrn_predictor.objs[number] == name:
                    obj_id = int(number) 
                    break
            assert obj_id > 0
            ymin = roi.x_offset
            xmin = roi.y_offset
            ymax = ymin + roi.width
            xmax = xmin + roi.height
            outputs = torch.tensor([float(ymin), float(xmin), float(ymax), float(xmax),  score, score, float(obj_id - 1)])
            outputs = list((outputs.unsqueeze(0)))

            data_dict = self.gdrn_predictor.preprocessing(outputs=outputs, image=image, depth_img=depth_img)
            out_dict = self.gdrn_predictor.inference(data_dict)
            poses = self.gdrn_predictor.postprocessing(
                data_dict,
                out_dict,
                self.renderer_request_queue, 
                self.renderer_result_queue)
            #self.gdrn_predictor.gdrn_visualization(batch=data_dict, out_dict=out_dict, image=image)

            obj_name = self.gdrn_predictor.objs[int(obj_id)]

            R_0 = np.eye(4,4)
            R = poses[obj_name][ 0:3,0:3 ]
            R_0[ 0:3,0:3 ] = R
            t = poses[obj_name][ 0:3,3:4 ].ravel()

            rot_quat = tf.transformations.quaternion_from_matrix(R_0)

            br = tf.TransformBroadcaster()
            br.sendTransform((poses[obj_name][0][3], poses[obj_name][1][3], poses[obj_name][2][3]),
                        rot_quat,
                        rospy.Time.now(),
                        f"pose_{obj_name}",
                        self.frame_id)

            confidence = score
            pose = Pose()
            pose.position.x = poses[obj_name][0][3]
            pose.position.y = poses[obj_name][1][3]
            pose.position.z = poses[obj_name][2][3]
            pose.orientation.x = rot_quat[0]
            pose.orientation.y = rot_quat[1]
            pose.orientation.z = rot_quat[2]
            pose.orientation.w = rot_quat[3]
            pose_results.append(pose)
            valid_class_names.append(name)
            class_confidences.append(confidence)
        
        response = GenericImgProcAnnotatorResult()
        response.pose_results = pose_results
        response.class_names = valid_class_names
        response.class_confidences = class_confidences

        end_time = time.time()
        elapsed_time = end_time - start_time
        #print('Execution time:', elapsed_time, 'seconds')
        self.server.set_succeeded(response)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ycbv', help='name of the dataset')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    renderer_request_queue = queue.Queue()
    renderer_result_queue = queue.Queue()

    GDRN_ROS(renderer_request_queue, renderer_result_queue, **vars(opt))
    
    # Load camera intrinsics from file
    intrinsics = np.asarray(rospy.get_param('/pose_estimator/intrinsics'))
    renderer = RendererROS((64, 64), intrinsics, model_paths=None, scale_to_meter=1.0, gpu_id=None)

    while not rospy.is_shutdown():
        if not renderer_request_queue.empty():
            request = renderer_request_queue.get(block=True, timeout=0.2)

            K_crop  = request[0]
            model = request[1]
            pose_est = request[2]
            renderer.clear() 
            renderer.set_cam(K_crop)
            renderer.draw_model(model,pose_est)
            _, ren_dp = renderer.finish()
            renderer_result_queue.put(ren_dp)
        else:
            rospy.sleep(0.1)