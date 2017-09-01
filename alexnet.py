import torch.nn as nn
import torchvision.transforms as transforms

__all__ = ['alexnet']

class AlexNetOWT_BN(nn.Module):

    def __init__(self, num_classes=1000, kernel_size1=11, stride1=4, kernel_size2=5,
                 kernel_size3=3, kernel_size4=3, kernel_size5=3):
        super(AlexNetOWT_BN, self).__init__()
        self.kernel_size1 = kernel_size1
        self.stride1 = stride1
        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernel_size3
        self.kernel_size4 = kernel_size4
        self.kernel_size5 = kernel_size5

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, self.kernel_size1, stride=self.stride1, padding=2,
                      bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=self.kernel_size2, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 384, kernel_size=self.kernel_size3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 256, kernel_size=self.kernel_size4, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=self.kernel_size5, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 1e-2,
                'weight_decay': 5e-4, 'momentum': 0.9},
            10: {'lr': 5e-3},
            15: {'lr': 1e-3, 'weight_decay': 0},
            20: {'lr': 5e-4},
            25: {'lr': 1e-4}
        }
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.input_transform = {
            'train': transforms.Compose([
                transforms.Scale(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'eval': transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        }

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(**kwargs):
    num_classes = getattr(kwargs, 'num_classes', 1000)
    return AlexNetOWT_BN(num_classes)
