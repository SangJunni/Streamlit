import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from efficientnet_pytorch import EfficientNet
import os
import yaml
os.environ['KMP_DUPLICATE_LIB_OK']='True'

with open("config.yaml") as f:
        config = yaml.load(f, Loader = yaml.FullLoader)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyEfficientNet(nn.Module) :
    '''
    EfiicientNet-b4의 출력층만 변경합니다.
    한번에 18개의 Class를 예측하는 형태의 Model입니다.
    '''
    def __init__(self, num_classes: int = 18) :
        super(MyEfficientNet, self).__init__()
        self.EFF = EfficientNet.from_pretrained('efficientnet-b4', in_channels=3, num_classes=num_classes)
    
    def forward(self, x) -> torch.Tensor:
        x = self.EFF(x)
        x = F.softmax(x, dim=1)
        return x

class MyResNet_mask(nn.Module) :
    '''
    EfiicientNet-b4의 출력층만 변경합니다.
    한번에 3개의 Mask Class를 예측하는 형태의 Model입니다.
    '''
    def __init__(self, num_classes: int = 3) :
        super(MyResNet_mask, self).__init__()
        Res = torchvision.models.resnet50(pretrained=True)
        Res.fc=nn.Linear(Res.fc.in_features, num_classes)
        Res.load_state_dict(torch.load(config['model_path_mask'], map_location = device))
        self.ResNet = Res
        
    def forward(self, x) -> torch.Tensor:
        x = self.ResNet(x)
        return x

class MyResNet_age(nn.Module) :
    '''
    EfiicientNet-b4의 출력층만 변경합니다.
    한번에 3개의 Age Class를 예측하는 형태의 Model입니다.
    '''
    def __init__(self, num_classes: int = 3) :
        super(MyResNet_age, self).__init__()
        Res = torchvision.models.resnet50(pretrained=True)
        Res.fc=nn.Linear(Res.fc.in_features, num_classes)
        Res.load_state_dict(torch.load(config['model_path_age'], map_location = device))
        self.ResNet = Res
        
    def forward(self, x) -> torch.Tensor:
        x = self.ResNet(x)
        return x

class MyResNet_gender(nn.Module) :
    '''
    Resnet50의 출력층만 변경합니다.
    한번에 2개의 Gender를 예측하는 형태의 Model입니다.
    '''
    def __init__(self, num_classes: int = 2) :
        super(MyResNet_gender, self).__init__()
        Res = torchvision.models.resnet50(pretrained=True)
        Res.fc=nn.Linear(Res.fc.in_features, num_classes)
        Res.load_state_dict(torch.load(config['model_path_gender'], map_location = device))
        self.ResNet = Res
        
    def forward(self, x) -> torch.Tensor:
        x = self.ResNet(x)
        return x