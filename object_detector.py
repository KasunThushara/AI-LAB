import numpy as np
import cv2
from base_models import DetectionModelBase
from hailo_platform import (HEF, ConfigureParams, FormatType, HailoSchedulingAlgorithm, 
                           HailoStreamInterface, InferVStreams, InputVStreamParams, 
                           InputVStreams, OutputVStreamParams, OutputVStreams, Device, VDevice)
from timeit import default_timer as timer

class ObjectDetector(DetectionModelBase):
    def __init__(self, model_type, hailo_infer):
        super().__init__()
        self.model_type = model_type
        self.hailo_infer = hailo_infer
        self.batch_size = 1

    def load_model(self, model_path):
        if self.DEBUG:
            print(f"[ObjectDetector.load_model] Loading: {model_path}")

        self.hef_id = self.hailo_infer.load_model(model_path)
        self.hef = self.hailo_infer.hef_list[self.hef_id]
        self.network_group = self.hailo_infer.network_group_list[self.hef_id]
        self.network_group_params = self.hailo_infer.network_group_params_list[self.hef_id]
        self.input_vstreams_params = self.hailo_infer.input_vstreams_params_list[self.hef_id]
        self.output_vstreams_params = self.hailo_infer.output_vstreams_params_list[self.hef_id]

        self.input_vstream_infos = self.hef.get_input_vstream_infos()
        self.output_vstream_infos = self.hef.get_output_vstream_infos()
        self.inputShape = tuple(self.input_vstream_infos[0].shape)

        if self.model_type == "palm":
            if len(self.output_vstream_infos) == 6:
                self.outputShape1, self.outputShape2 = (1,2944,1), (1,2944,18)
            elif len(self.output_vstream_infos) == 4:
                self.outputShape1, self.outputShape2 = (1,2016,1), (1,2016,18)
        elif self.model_type == "face":
            if len(self.output_vstream_infos) == 4:
                self.outputShape1, self.outputShape2 = (1,896,1), (1,896,16)
            elif len(self.output_vstream_infos) == 2:
                self.outputShape1, self.outputShape2 = (1,2304,1), (1,2304,16)

        self.x_scale = self.y_scale = self.inputShape[1]
        self.h_scale = self.w_scale = self.inputShape[2]
        self.num_anchors = self.outputShape2[1]
        
        self.config_model(self.model_type)

    def preprocess(self, x):
        return x.astype(np.uint8)

    def predict_on_image(self, img):
        img_expanded = np.expand_dims(img, axis=0)
        detections = self.predict_on_batch(img_expanded)
        return np.array(detections)[0] if len(detections) > 0 else []

    def predict_on_batch(self, x):
        self.profile_pre = self.profile_model = self.profile_post = 0.0

        start = timer()
        x = self.preprocess(x)
        input_data = {self.input_vstream_infos[0].name: x}
        self.profile_pre = timer() - start

        start = timer()
        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                infer_results = infer_pipeline.infer(input_data)
        self.profile_model = timer() - start

        start = timer()
        if self.model_type == "palm":
            if len(self.output_vstream_infos) == 6:
                out1 = np.concatenate([
                    infer_results[self.output_vstream_infos[2].name].reshape(1,384,1),
                    infer_results[self.output_vstream_infos[1].name].reshape(1,512,1),
                    infer_results[self.output_vstream_infos[0].name].reshape(1,2048,1)
                ], axis=1).astype(np.float32)
                out2 = np.concatenate([
                    infer_results[self.output_vstream_infos[5].name].reshape(1,384,18),
                    infer_results[self.output_vstream_infos[4].name].reshape(1,512,18),
                    infer_results[self.output_vstream_infos[3].name].reshape(1,2048,18)
                ], axis=1).astype(np.float32)
            elif len(self.output_vstream_infos) == 4:
                out1 = np.concatenate([
                    infer_results[self.output_vstream_infos[1].name].reshape(1,1152,1),
                    infer_results[self.output_vstream_infos[0].name].reshape(1,864,1)
                ], axis=1).astype(np.float32)
                out2 = np.concatenate([
                    infer_results[self.output_vstream_infos[3].name].reshape(1,1152,18),
                    infer_results[self.output_vstream_infos[2].name].reshape(1,864,18)
                ], axis=1).astype(np.float32)
        elif self.model_type == "face":
            if len(self.output_vstream_infos) == 4:
                out1 = np.concatenate([
                    infer_results[self.output_vstream_infos[3].name].reshape(1,512,1),
                    infer_results[self.output_vstream_infos[2].name].reshape(1,384,1)
                ], axis=1).astype(np.float32)
                out2 = np.concatenate([
                    infer_results[self.output_vstream_infos[1].name].reshape(1,512,16),
                    infer_results[self.output_vstream_infos[0].name].reshape(1,384,16)
                ], axis=1).astype(np.float32)
            elif len(self.output_vstream_infos) == 2:
                out1 = infer_results[self.output_vstream_infos[1].name].reshape(1,2304,1).astype(np.float32)
                out2 = infer_results[self.output_vstream_infos[0].name].reshape(1,2304,16).astype(np.float32)

        detections = self._tensors_to_detections(out2, out1, self.anchors)
        filtered_detections = [self._weighted_non_max_suppression(d) for d in detections if len(d) > 0]
        
        self.profile_post = timer() - start
        return [d for d in filtered_detections if len(d) > 0]