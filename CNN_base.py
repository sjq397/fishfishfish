import torch
import torch.nn as nn

class Conv1DModel(nn.Module):
    def __init__(self, num_classes, audio_length):
        super(Conv1DModel, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=80, stride=4, padding=40)  
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)  
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        self.dropout2 = nn.Dropout(0.5)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=4)
        self.dropout3 = nn.Dropout(0.3)

        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1) 
        self.bn4 = nn.BatchNorm1d(64)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=4)

        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout4 = nn.Dropout(0.5)

        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = self.global_pooling(x)  # Global average pooling for 1D data
        x = x.squeeze(-1)  # Remove the last dimension after pooling
        x = self.dropout4(x)
        
        x = self.fc(x)
        return x


# model = Conv1DModel(num_classes=2, audio_length=5000)  
