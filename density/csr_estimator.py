import torch
import numpy as np
import cv2
from .csrnet import CSRNet
from PIL import Image
from torchvision import transforms

class CSREstimator:
    def __init__(self, model_path=None, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CSRNet(load_weights=(model_path is not None))
        
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def estimate(self, frame):
        # Convert OpenCV BGR to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
        
        # Output is a density map
        density_map = output.detach().cpu().numpy().squeeze()
        count = np.sum(density_map)
        
        return density_map, count
