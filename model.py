import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        
        # Define layers for the first input image
        self.conv1_image1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1_image1 = nn.ReLU()
        
        # Define layers for the second input image
        self.conv1_image2 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1_image2 = nn.ReLU()
        
        # Define layers for combining the features from both images
        self.conv_combined = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu_combined = nn.ReLU()
        
        # Define layers for the final output image
        self.conv_output = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
    def forward(self, image1, image2):
        features_image1 = self.relu1_image1(self.conv1_image1(image1))
        features_image2 = self.relu1_image2(self.conv1_image2(image2))
        
        # Concatenate the features from both images along the channel dimension
        combined_features = torch.cat((features_image1, features_image2), dim=1)
        
        combined_features = self.relu_combined(self.conv_combined(combined_features))
        output_image = self.conv_output(combined_features)
        
        return output_image

