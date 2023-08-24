import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

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
        
    def forward(self, image):
        features_image1 = self.relu1_image1(self.conv1_image1(image[:,0:3]))
        features_image2 = self.relu1_image2(self.conv1_image2(image[:,3:6]))
        
        # Concatenate the features from both images along the channel dimension
        combined_features = torch.cat((features_image1, features_image2), dim=1)
        
        combined_features = self.relu_combined(self.conv_combined(combined_features))
        output_image = self.conv_output(combined_features)
        
        return output_image

class fcn(nn.Module):
    def __init__(self, n_classes=3):
        super(fcn, self).__init__()
        self.n_classes = n_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )


        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2048, 3, kernel_size=3, padding=1)
            
        )


    def forward(self, x):
        conv1_1 = self.conv_block1(x[:,0:3])
        conv1_2 = self.conv_block1(x[:,3:6])
        conv2_1 = self.conv_block2(conv1_1)
        conv2_2 = self.conv_block2(conv1_2)
        conv3_1 = self.conv_block3(conv2_1)
        conv3_2 = self.conv_block3(conv2_2)
        conv4_1 = self.conv_block4(conv3_1)
        conv4_2 = self.conv_block4(conv3_2)
        conv5_1 = self.conv_block5(conv4_1)
        conv5_2 = self.conv_block5(conv4_2)

        # Concatenate the features from both images along the channel dimension
        combined_features = torch.cat((conv5_1, conv5_2), dim=1)


        score = self.classifier(combined_features)
        out = F.interpolate(score, size=x.size()[2:], mode='bilinear', align_corners=False)

        return out

