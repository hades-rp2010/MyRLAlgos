import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import cv2

class LinearQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNQNetwork(nn.Module):
    def __init__(self, output_dim, input_channels=4):
        super(CNNQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Compute the size after convolutions for a 84x84 input
        # After conv1: (84-8)/4 + 1 = 20
        # After conv2: (20-4)/2 + 1 = 9
        # After conv3: (9-3)/1 + 1 = 7
        # So 64 * 7 * 7 = 3136
        
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, output_dim)
    
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FramePreprocessor:
    """Preprocesses and stacks frames for Atari games"""
    def __init__(self, frame_stack_size=4, target_size=84):
        self.frame_stack_size = frame_stack_size
        self.target_size = target_size
        self.frame_buffer = deque(maxlen=frame_stack_size)
    
    def preprocess_frame(self, frame):
        """Convert RGB frame to grayscale and resize"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize to target size
        resized = cv2.resize(gray, (self.target_size, self.target_size))
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        return normalized
    
    def reset(self):
        """Clear the frame buffer"""
        self.frame_buffer.clear()
    
    def add_frame(self, frame):
        """Add a preprocessed frame to the buffer"""
        preprocessed = self.preprocess_frame(frame)
        self.frame_buffer.append(preprocessed)
    
    def get_stacked_frames(self):
        """Return stacked frames as (channels, height, width) tensor"""
        # If buffer not full, pad with first frame
        if len(self.frame_buffer) < self.frame_stack_size:
            frames = list(self.frame_buffer)
            while len(frames) < self.frame_stack_size:
                frames.insert(0, frames[0] if frames else np.zeros((self.target_size, self.target_size)))
        else:
            frames = list(self.frame_buffer)
        
        # Stack frames along channel dimension
        stacked = np.stack(frames, axis=0)  # Shape: (frame_stack_size, target_size, target_size)
        return torch.tensor(stacked, dtype=torch.float32)
