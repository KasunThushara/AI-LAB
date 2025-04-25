import numpy as np
import cv2
from base_models import LandmarkModelBase
from hailo_platform import (HEF, ConfigureParams, FormatType, HailoSchedulingAlgorithm,
                           HailoStreamInterface, InferVStreams, InputVStreamParams,
                           OutputVStreamParams, Device, VDevice)
from timeit import default_timer as timer

class LandmarkPredictor(LandmarkModelBase):
    def __init__(self, model_type, hailo_infer):
        super(LandmarkPredictor, self).__init__()
        self.model_type = model_type
        self.hailo_infer = hailo_infer

    def load_model(self, model_path):
        if self.DEBUG:
            print("[LandmarkPredictor.load_model] Model File:", model_path)

        self.hef_id = self.hailo_infer.load_model(model_path)
        self.hef = self.hailo_infer.hef_list[self.hef_id]
        self.network_group = self.hailo_infer.network_group_list[self.hef_id]
        self.network_group_params = self.hailo_infer.network_group_params_list[self.hef_id]
        self.input_vstreams_params = self.hailo_infer.input_vstreams_params_list[self.hef_id]
        self.output_vstreams_params = self.hailo_infer.output_vstreams_params_list[self.hef_id]

        self.input_vstream_infos = self.hef.get_input_vstream_infos()
        self.output_vstream_infos = self.hef.get_output_vstream_infos()
        self.num_inputs = len(self.input_vstream_infos)
        self.num_outputs = len(self.output_vstream_infos)

        self.inputShape = self.input_vstream_infos[0].shape
        
        if self.model_type == "hand":
            if self.inputShape[1] == 224:  # hand_landmark_v0_07
                self.outputShape1 = (1, 1)
                self.outputShape2 = (1, 63)
            else:  # hand_landmark_lite/full
                self.outputShape1 = tuple(self.output_vstream_infos[2].shape)
                self.outputShape2 = tuple(self.output_vstream_infos[0].shape)
        elif self.model_type == "face":
            self.outputShape1 = tuple(self.output_vstream_infos[0].shape)
            self.outputShape2 = tuple(self.output_vstream_infos[1].shape)
        elif self.model_type == "pose":
            self.outputShape1 = tuple(self.output_vstream_infos[0].shape)
            self.outputShape2 = tuple(self.output_vstream_infos[1].shape)

        self.resolution = self.inputShape[1]
        if self.DEBUG:
            print(f"[LandmarkPredictor.load_model] Input Resolution: {self.resolution}")

    def preprocess(self, x):
        return (x * 255.0).astype(np.uint8)

    def predict(self, x):
        self.profile_pre = 0.0
        self.profile_model = 0.0
        self.profile_post = 0.0

        flag_list = []
        landmarks_list = []
        nb_images = x.shape[0]

        start = timer()
        x = self.preprocess(x)
        self.profile_pre += timer() - start

        for i in range(nb_images):
            start = timer()
            image_input = np.expand_dims(x[i, :, :, :], axis=0)
            input_data = {self.input_vstream_infos[0].name: image_input}
            self.profile_pre += timer() - start

            start = timer()
            with InferVStreams(self.network_group, self.input_vstreams_params, 
                             self.output_vstreams_params) as infer_pipeline:
                with self.network_group.activate(self.network_group_params):
                    infer_results = infer_pipeline.infer(input_data)
            self.profile_model += timer() - start

            start = timer()
            if self.model_type == "hand":
                if self.resolution == 256:
                    flag = infer_results[self.output_vstream_infos[1].name].reshape(1, 1)
                    landmarks = infer_results[self.output_vstream_infos[0].name].reshape(1, 21, -1)
                else:
                    flag = infer_results[self.output_vstream_infos[2].name]
                    landmarks = infer_results[self.output_vstream_infos[0].name].reshape(1, 21, -1)
                landmarks = landmarks / self.resolution
            elif self.model_type == "face":
                flag = infer_results[self.output_vstream_infos[0].name]
                landmarks = infer_results[self.output_vstream_infos[1].name].reshape(1, -1, 3)
                landmarks = landmarks / self.resolution
            elif self.model_type == "pose":
                flag = infer_results[self.output_vstream_infos[1].name]
                landmarks = infer_results[self.output_vstream_infos[0].name].reshape(1, -1, 5)
                landmarks = landmarks / self.resolution

            flag_list.append(flag.squeeze(0))
            landmarks_list.append(landmarks.squeeze(0))
            self.profile_post += timer() - start

        return np.asarray(flag_list), np.asarray(landmarks_list)