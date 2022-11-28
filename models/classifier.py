import torch
from torchvision.models import resnet50, ResNet50_Weights


class CNNClassifier(torch.nn.Module):
    def __init__(self, n_class: int, freeze_cnn: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=weights)
        self.model.fc = torch.nn.Linear(2048, n_class)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        if freeze_cnn:
            for name, param in self.model.named_parameters():
                if param.requires_grad and 'fc' not in name:
                    param.requires_grad = False

    def forward(self, img):
        out = self.model(img)
        return self.log_softmax(out)


class KeypointClassifier(torch.nn.Module):
    def __init__(self, input_size: int, n_class: int):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Linear(64, n_class),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    model = CNNClassifier(7)
