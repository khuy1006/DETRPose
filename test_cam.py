import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from copy import deepcopy
import time

# Add DETRPose to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core import LazyConfig, instantiate
from tools.inference.annotators import COCOVisualizer, CrowdPoseVisualizer


def load_model(model_name='m'):
    """Load DETRPose model"""
    config_models = {
        # For COCO2017
        'n': 'configs/detrpose/detrpose_hgnetv2_n.py',
        's': 'configs/detrpose/detrpose_hgnetv2_s.py',
        'm': 'configs/detrpose/detrpose_hgnetv2_m.py',
        'l': 'configs/detrpose/detrpose_hgnetv2_l.py',
        'x': 'configs/detrpose/detrpose_hgnetv2_x.py',
        # For CrowdPose
        'n_crowdpose': 'configs/detrpose/detrpose_hgnetv2_n_crowdpose.py',
        's_crowdpose': 'configs/detrpose/detrpose_hgnetv2_s_crowdpose.py',
        'm_crowdpose': 'configs/detrpose/detrpose_hgnetv2_m_crowdpose.py',
        'l_crowdpose': 'configs/detrpose/detrpose_hgnetv2_l_crowdpose.py',
        'x_crowdpose': 'configs/detrpose/detrpose_hgnetv2_x_crowdpose.py',
    }
    
    config_path = config_models[model_name]
    download_url = f'https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_{model_name}.pth'
    weights_folder = 'models/detrpose/'
    
    # Determine which annotator to use
    if 'crowdpose' in model_name:
        Drawer = CrowdPoseVisualizer
    else:
        Drawer = COCOVisualizer
    
    # Load configuration
    cfg = LazyConfig.load(config_path)
    if hasattr(cfg.model.backbone, 'pretrained'):
        cfg.model.backbone.pretrained = False
    
    # Load model weights
    print(f"Loading model weights for {model_name}...")
    state = torch.hub.load_state_dict_from_url(download_url, map_location='cpu', model_dir=weights_folder)
    model = instantiate(cfg.model)
    postprocessor = instantiate(cfg.postprocessor)
    
    checkpoint = torch.load(f"{weights_folder}/detrpose_hgnetv2_{model_name}.pth", map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)
    
    # Create wrapper model
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model.deploy()
            self.postprocessor = postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    return Model(), Drawer


def main():
    # Configuration
    model_name = 'n'  # Options: 'n', 's', 'm', 'l', 'x' or add '_crowdpose' suffix
    camera_id = 0  # 0 for default camera, or use video file path
    threshold = 0.5  # Detection threshold
    
    # Load model
    model, Drawer = load_model(model_name)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    # Setup transforms
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_id)
    
    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Press 'q' to quit")
    fps_time = time.time()
    
    # Initialize visualizer
    visualizer = Drawer()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Get image dimensions
            h, w = frame.shape[:2]
            orig_size = torch.tensor([[w, h]]).to(device)
            
            # Convert frame to PIL for transforms
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(frame_rgb)
            
            # Prepare input
            im_data = transforms(im_pil).unsqueeze(0).to(device)
            
            # Run inference
            with torch.no_grad():
                output = model(im_data, orig_size)
            
            scores, labels, keypoints = output
            scores, labels, keypoints = scores[0], labels[0], keypoints[0]
            
            # Filter keypoints by threshold
            valid_mask = scores > threshold
            valid_keypoints = keypoints[valid_mask].cpu().numpy()
            
            # Draw keypoints on frame
            if len(valid_keypoints) > 0:
                frame = visualizer.draw_on(frame, valid_keypoints)
            
            # Calculate and display FPS
            fps = 1.0 / (time.time() - fps_time)
            fps_time = time.time()
            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('DETRPose - Camera Feed', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released")


if __name__ == "__main__":
    main()