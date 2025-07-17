
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class YOLOv2MobileNet(nn.Module):
    """
    Модель YOLOv2 на базе MobileNetV2 в качестве feature extractor.
    """

    def __init__(self, S=16, B=2, C=8):
        super(YOLOv2MobileNet, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        out_channels = B * (5 + C)

        # Загружаем MobileNetV2 без последнего классификатора
        mobilenet = models.mobilenet_v2(pretrained=True)

        self.features = mobilenet.features  # только сверточные слои MobileNet

        self.head = nn.Sequential(
            nn.Conv2d(1280, 512, 3, padding=1),  # MobileNetV2  1280 каналов
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Conv2d(512, out_channels, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        x = x.permute(0, 2, 3, 1)

        B, S, _, _ = x.shape
        x = x.view(B, self.S, self.S, self.B, 5 + self.C)

        x[..., 0:2] = torch.sigmoid(x[..., 0:2])  # x, y
        x[..., 2:4] = torch.sigmoid(x[..., 2:4])  # w, h
        x[..., 4] = torch.sigmoid(x[..., 4])      # уверенность
        x[..., 5:] = torch.sigmoid(x[..., 5:])    # вероятность класса

        return x.view(B, self.S, self.S, -1)

    def initialize_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
