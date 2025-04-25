import sys
import os
import numpy as np
import cv2
import argparse
import socket
import pickle
import struct

# Import Hailo-specific modules
from hailo_inference import HailoInference
from object_detector import ObjectDetector  
from landmark_predictor import LandmarkPredictor  
from visualization import draw_detections, draw_landmarks, draw_roi
from visualization import HAND_CONNECTIONS, FACE_CONNECTIONS

class NetworkStreamer:
    def __init__(self, host, port):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
        self.server_address = (host, port)
        
    def send_frame(self, frame, quality=50):
        try:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ret:
                return False
                
            # Serialize and send
            data = pickle.dumps(buffer)
            self.client_socket.sendto(data, self.server_address)
            return True
        except Exception as e:
            print(f"Network error: {str(e)}")
            return False

def main():
    # Initialize Hailo inference
    hailo_infer = HailoInference()

    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--detection', type=str, default="hand",  
                    choices=["hand", "face"],
                    help="Application (hand, face). Default is hand")
    ap.add_argument('-m', '--model1', type=str, 
                    help='Path of detection model')
    ap.add_argument('-n', '--model2', type=str, 
                    help='Path of landmark model')
    ap.add_argument('--host', type=str, default="127.0.0.1",
                    help='Server IP address')
    ap.add_argument('--port', type=int, default=9999,
                    help='Server port number')
    args = ap.parse_args()

    # Set up models based on application type
    if args.detection == "hand":  
        detector_type = "palm"  
        landmark_type = "hand"  
        default_detector_model = 'models/palm_detection_lite.hef'
        default_landmark_model = 'models/hand_landmark_lite.hef'
    elif args.detection == "face":  
        detector_type = "face"  
        landmark_type = "face"  
        default_detector_model = 'models/face_detection_short_range.hef'
        default_landmark_model = 'models/face_landmark.hef'
    else:
        print(f"[ERROR] Invalid application: {args.detection}. Must be one of hand,face.")  # Updated error message
        exit(1)

    # Use default models if none specified
    args.model1 = args.model1 or default_detector_model
    args.model2 = args.model2 or default_landmark_model

    # Initialize detectors
    detector = ObjectDetector(detector_type, hailo_infer)  
    detector.load_model(args.model1)

    landmark_predictor = LandmarkPredictor(landmark_type, hailo_infer)  
    landmark_predictor.load_model(args.model2)

    # Initialize network streamer
    streamer = NetworkStreamer(args.host, args.port)

    # Open default camera
    cap = cv2.VideoCapture(0)
    frame_width, frame_height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    print(f"Streaming {args.detection} detection to {args.host}:{args.port}")  
    print("Press Ctrl+C to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Convert frame to RGB and process
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = frame.copy()
            
            # Detection pipeline
            img1, scale1, pad1 = detector.resize_pad(image)  
            normalized_detections = detector.predict_on_image(img1)  
            
            if len(normalized_detections) > 0:
                detections = detector.denormalize_detections(normalized_detections, scale1, pad1)  # Changed
                
                # Get ROI for landmarks
                xc, yc, scale, theta = detector.detection2roi(detections)  # Changed
                roi_img, roi_affine, roi_box = landmark_predictor.extract_roi(image, xc, yc, theta, scale)  # Changed
                
                # Get landmarks
                flags, normalized_landmarks = landmark_predictor.predict(roi_img)  # Changed
                landmarks = landmark_predictor.denormalize_landmarks(normalized_landmarks, roi_affine)  # Changed
                
                # Draw results on output frame
                for i in range(len(flags)):
                    landmark, flag = landmarks[i], flags[i]
                    if args.detection == "hand":  
                        draw_landmarks(output, landmark[:, :2], HAND_CONNECTIONS, size=2)
                    elif args.detection == "face":  
                        draw_landmarks(output, landmark[:, :2], FACE_CONNECTIONS, size=1)
                
                draw_roi(output, roi_box)
                draw_detections(output, detections)
            
            # Send frame over network instead of displaying
            if not streamer.send_frame(output):
                print("Frame transmission failed")
                break

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        cap.release()
        print("Stream ended")

if __name__ == "__main__":
    main()