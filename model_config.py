import numpy as np

# Anchor options references:
# https://github.com/hollance/BlazeFace-PyTorch/blob/master/Anchors.ipynb
# https://github.com/google/mediapipe/

# Palm detection configurations
palm_detect_v0_06_anchor_options = {
    "num_layers": 5,
    "min_scale": 0.1171875,
    "max_scale": 0.75,
    "input_size_height": 256,
    "input_size_width": 256,
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [8, 16, 32, 32, 32],
    "aspect_ratios": [1.0],
    "reduce_boxes_in_lowest_layer": False,
    "interpolated_scale_aspect_ratio": 1.0,
    "fixed_anchor_size": True,
}

palm_detect_v0_06_model_config = {    
    "num_classes": 1,
    "num_anchors": 2944,
    "num_coords": 18,
    "score_clipping_thresh": 100.0,
    "x_scale": 256.0,
    "y_scale": 256.0,
    "h_scale": 256.0,
    "w_scale": 256.0,
    "min_score_thresh": 0.7,
    "min_suppression_threshold": 0.3,
    "num_keypoints": 7,
    "detection2roi_method": 'box',
    "kp1": 0,
    "kp2": 2,
    "theta0": np.pi/2,
    "dscale": 2.6,
    "dy": -0.5,
}

palm_detect_v0_10_anchor_options = {
    "num_layers": 4,
    "min_scale": 0.1484375,
    "max_scale": 0.75,
    "input_size_height": 192,
    "input_size_width": 192,
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [8, 16, 16, 16],
    "aspect_ratios": [1.0],
    "reduce_boxes_in_lowest_layer": False,
    "interpolated_scale_aspect_ratio": 1.0,
    "fixed_anchor_size": True,
}

palm_detect_v0_10_model_config = {    
    "num_classes": 1,
    "num_anchors": 2016,
    "num_coords": 18,
    "score_clipping_thresh": 100.0,
    "x_scale": 192.0,
    "y_scale": 192.0,
    "h_scale": 192.0,
    "w_scale": 192.0,
    "min_score_thresh": 0.5,
    "min_suppression_threshold": 0.3,
    "num_keypoints": 7,
    "detection2roi_method": 'box',
    "kp1": 0,
    "kp2": 2,
    "theta0": np.pi/2,
    "dscale": 2.6,
    "dy": -0.5,
}

# Face detection configurations
face_front_v0_06_anchor_options = {
    "num_layers": 4,
    "min_scale": 0.1484375,
    "max_scale": 0.75,
    "input_size_height": 128,
    "input_size_width": 128,
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [8, 16, 16, 16],
    "aspect_ratios": [1.0],
    "reduce_boxes_in_lowest_layer": False,
    "interpolated_scale_aspect_ratio": 1.0,
    "fixed_anchor_size": True,
}

face_front_v0_06_model_config = {    
    "num_classes": 1,
    "num_anchors": 896,
    "num_coords": 16,
    "score_clipping_thresh": 100.0,
    "x_scale": 128.0,
    "y_scale": 128.0,
    "h_scale": 128.0,
    "w_scale": 128.0,
    "min_score_thresh": 0.75,
    "min_suppression_threshold": 0.3,
    "num_keypoints": 6,
    "detection2roi_method": 'box',
    "kp1": 1,
    "kp2": 0,
    "theta0": 0.,
    "dscale": 1.5,
    "dy": 0.,
}

face_back_v0_07_anchor_options = {
    "num_layers": 4,
    "min_scale": 0.15625,
    "max_scale": 0.75,
    "input_size_height": 256,
    "input_size_width": 256,
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [16, 32, 32, 32],
    "aspect_ratios": [1.0],
    "reduce_boxes_in_lowest_layer": False,
    "interpolated_scale_aspect_ratio": 1.0,
    "fixed_anchor_size": True,
}

face_back_v0_07_model_config = {    
    "num_classes": 1,
    "num_anchors": 896,
    "num_coords": 16,
    "score_clipping_thresh": 100.0,
    "x_scale": 256.0,
    "y_scale": 256.0,
    "h_scale": 256.0,
    "w_scale": 256.0,
    "min_score_thresh": 0.65,
    "min_suppression_threshold": 0.3,
    "num_keypoints": 6,
    "detection2roi_method": 'box',
    "kp1": 1,
    "kp2": 0,
    "theta0": 0.,
    "dscale": 1.5,
    "dy": 0.,
}

face_short_range_v0_10_anchor_options = {
    "num_layers": 4,
    "min_scale": 0.1484375,
    "max_scale": 0.75,
    "input_size_height": 128,
    "input_size_width": 128,
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [8, 16, 16, 16],
    "aspect_ratios": [1.0],
    "reduce_boxes_in_lowest_layer": False,
    "interpolated_scale_aspect_ratio": 1.0,
    "fixed_anchor_size": True,
}

face_short_range_v0_10_model_config = {    
    "num_classes": 1,
    "num_anchors": 896,
    "num_coords": 16,
    "score_clipping_thresh": 100.0,
    "x_scale": 128.0,
    "y_scale": 128.0,
    "h_scale": 128.0,
    "w_scale": 128.0,
    "min_score_thresh": 0.5,
    "min_suppression_threshold": 0.3,
    "num_keypoints": 6,
    "detection2roi_method": 'box',
    "kp1": 1,
    "kp2": 0,
    "theta0": 0.,
    "dscale": 1.5,
    "dy": 0.,
}

face_full_range_v0_10_anchor_options = {
    "num_layers": 1,
    "min_scale": 0.1484375,
    "max_scale": 0.75,
    "input_size_height": 192,
    "input_size_width": 192,
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [4],
    "aspect_ratios": [1.0],
    "reduce_boxes_in_lowest_layer": False,
    "interpolated_scale_aspect_ratio": 0.0,
    "fixed_anchor_size": True,
}

face_full_range_v0_10_model_config = {    
    "num_classes": 1,
    "num_anchors": 2304,
    "num_coords": 16,
    "score_clipping_thresh": 100.0,
    "x_scale": 192.0,
    "y_scale": 192.0,
    "h_scale": 192.0,
    "w_scale": 192.0,
    "min_score_thresh": 0.6,
    "min_suppression_threshold": 0.3,
    "num_keypoints": 6,
    "detection2roi_method": 'box',
    "kp1": 1,
    "kp2": 0,
    "theta0": 0.,
    "dscale": 1.5,
    "dy": 0.,
}

# Pose detection configurations
pose_detect_v0_07_anchor_options = {
    "num_layers": 4,
    "min_scale": 0.1484375,
    "max_scale": 0.75,
    "input_size_height": 128,
    "input_size_width": 128,
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [8, 16, 16, 16],
    "aspect_ratios": [1.0],
    "reduce_boxes_in_lowest_layer": False,
    "interpolated_scale_aspect_ratio": 1.0,
    "fixed_anchor_size": True,
}

pose_detect_v0_07_model_config = {    
    "num_classes": 1,
    "num_anchors": 896,
    "num_coords": 12,
    "score_clipping_thresh": 100.0,
    "x_scale": 128.0,
    "y_scale": 128.0,
    "h_scale": 128.0,
    "w_scale": 128.0,
    "min_score_thresh": 0.5,
    "min_suppression_threshold": 0.3,
    "num_keypoints": 4,
    "detection2roi_method": 'alignment',
    "kp1": 2,
    "kp2": 3,
    "theta0": 90 * np.pi / 180,
    "dscale": 1.5,
    "dy": 0.,
}

pose_detect_v0_10_anchor_options = {
    "num_layers": 5,
    "min_scale": 0.1484375,
    "max_scale": 0.75,
    "input_size_height": 224,
    "input_size_width": 224,
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [8, 16, 32, 32, 32],
    "aspect_ratios": [1.0],
    "reduce_boxes_in_lowest_layer": False,
    "interpolated_scale_aspect_ratio": 1.0,
    "fixed_anchor_size": True,
}

pose_detect_v0_10_model_config = {    
    "num_classes": 1,
    "num_anchors": 2254,
    "num_coords": 12,
    "score_clipping_thresh": 100.0,
    "x_scale": 224.0,
    "y_scale": 224.0,
    "h_scale": 224.0,
    "w_scale": 224.0,
    "min_score_thresh": 0.5,
    "min_suppression_threshold": 0.3,
    "num_keypoints": 4,
    "detection2roi_method": 'alignment',
    "kp1": 2,
    "kp2": 3,
    "theta0": 90 * np.pi / 180,
    "dscale": 1.5,
    "dy": 0.,
}


def get_model_config(model_type, input_width, input_height, num_anchors):
    if model_type == "palm":
        if num_anchors == 2944 and input_width == 256:
            return palm_detect_v0_06_model_config
        elif num_anchors == 2016 and input_width == 192:
            return palm_detect_v0_10_model_config
    elif model_type == "face":
        if num_anchors == 896 and input_width == 128:
            return face_front_v0_06_model_config
        elif num_anchors == 896 and input_width == 256:
            return face_back_v0_07_model_config
        elif num_anchors == 2304 and input_width == 192:
            return face_full_range_v0_10_model_config
    elif model_type == "pose":       
        if num_anchors == 896 and input_width == 128:
            return pose_detect_v0_07_model_config
        elif num_anchors == 2254 and input_width == 224:
            return pose_detect_v0_10_model_config
    
    print(f"[ModelConfig.get_model_config] ERROR: Unsupported configuration - Type: {model_type}, Anchors: {num_anchors}, Size: {input_width}x{input_height}")
    return None
    
def get_anchor_options(model_type, input_width, input_height, num_anchors):
    if model_type == "palm":
        if num_anchors == 2944 and input_width == 256:
            return palm_detect_v0_06_anchor_options
        elif num_anchors == 2016 and input_width == 192:
            return palm_detect_v0_10_anchor_options
    elif model_type == "face":
        if num_anchors == 896 and input_width == 128:
            return face_front_v0_06_anchor_options
        elif num_anchors == 896 and input_width == 256:
            return face_back_v0_07_anchor_options
        elif num_anchors == 2304 and input_width == 192:
            return face_full_range_v0_10_anchor_options
    elif model_type == "pose":       
        if num_anchors == 896 and input_width == 128:
            return pose_detect_v0_07_anchor_options
        elif num_anchors == 2254 and input_width == 224:
            return pose_detect_v0_10_anchor_options
    
    print(f"[ModelConfig.get_anchor_options] ERROR: Unsupported configuration - Type: {model_type}, Anchors: {num_anchors}, Size: {input_width}x{input_height}")
    return None


def calculate_scale(min_scale, max_scale, stride_index, num_strides):
    """Calculate scale for anchor generation."""
    if num_strides == 1:
        return (max_scale + min_scale) * 0.5
    return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1.0)


def generate_anchors(options):
    """Generate anchor boxes for object detection.
    
    Based on SSD anchor generation from:
    https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/ssd_anchors_calculator.cc
    """
    strides_size = len(options["strides"])
    assert options["num_layers"] == strides_size

    anchors = []
    layer_id = 0
    
    while layer_id < strides_size:
        anchor_height = []
        anchor_width = []
        aspect_ratios = []
        scales = []

        # For same strides, merge anchors in the same order
        last_same_stride_layer = layer_id
        while (last_same_stride_layer < strides_size and 
               options["strides"][last_same_stride_layer] == options["strides"][layer_id]):
            
            scale = calculate_scale(
                options["min_scale"],
                options["max_scale"],
                last_same_stride_layer,
                strides_size
            )

            if last_same_stride_layer == 0 and options["reduce_boxes_in_lowest_layer"]:
                # Special case for first layer
                aspect_ratios.extend([1.0, 2.0, 0.5])
                scales.extend([0.1, scale, scale])                
            else:
                for aspect_ratio in options["aspect_ratios"]:
                    aspect_ratios.append(aspect_ratio)
                    scales.append(scale)

                if options["interpolated_scale_aspect_ratio"] > 0.0:
                    scale_next = (1.0 if last_same_stride_layer == strides_size - 1
                                 else calculate_scale(
                                     options["min_scale"],
                                     options["max_scale"],
                                     last_same_stride_layer + 1,
                                     strides_size))
                    scales.append(np.sqrt(scale * scale_next))
                    aspect_ratios.append(options["interpolated_scale_aspect_ratio"])

            last_same_stride_layer += 1

        # Calculate anchor dimensions
        for i in range(len(aspect_ratios)):
            ratio_sqrts = np.sqrt(aspect_ratios[i])
            anchor_height.append(scales[i] / ratio_sqrts)
            anchor_width.append(scales[i] * ratio_sqrts)            
            
        # Generate grid of anchors
        stride = options["strides"][layer_id]
        feature_map_height = int(np.ceil(options["input_size_height"] / stride))
        feature_map_width = int(np.ceil(options["input_size_width"] / stride))

        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for anchor_id in range(len(anchor_height)):
                    x_center = (x + options["anchor_offset_x"]) / feature_map_width
                    y_center = (y + options["anchor_offset_y"]) / feature_map_height

                    new_anchor = [x_center, y_center, 0, 0]
                    if options["fixed_anchor_size"]:
                        new_anchor[2] = 1.0
                        new_anchor[3] = 1.0
                    else:
                        new_anchor[2] = anchor_width[anchor_id]
                        new_anchor[3] = anchor_height[anchor_id]
                    anchors.append(new_anchor)

        layer_id = last_same_stride_layer

    return np.asarray(anchors)