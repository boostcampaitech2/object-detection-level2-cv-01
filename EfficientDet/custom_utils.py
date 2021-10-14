
import torch

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet

# loss 추적
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

# Effdet config를 통해 모델 불러오기
def get_net(checkpoint_path=None):
    
    config = get_efficientdet_config('tf_efficientdet_d7')
    config.num_classes = 10
    config.image_size = (512, 512)
    
    config.soft_nms = False
    config.max_det_per_image = 25
    
    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        
    return DetBenchTrain(net)


# Effdet config를 통해 모델 불러오기 + ckpt load
def load_net(checkpoint_path, device):
    config = get_efficientdet_config('tf_efficientdet_d7')
    config.num_classes = 10
    config.image_size = (512,512)
    
    config.soft_nms = False
    config.max_det_per_image = 25
    
    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    net = DetBenchPredict(net)
    net.load_state_dict(checkpoint)
    net.eval()

    return net.to(device)
