import torch
from torchvision.models import resnet50, ResNet50_Weights
from models.keypoint_extractor import CustomKeypointExtractor


class CNNClassifier(torch.nn.Module):
    def __init__(self, n_class: int, freeze_cnn: bool = False) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=weights)
        self.channel_expander = torch.nn.Conv2d(1, 3, kernel_size=1)
        self.model.fc = torch.nn.Linear(2048, n_class)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        if freeze_cnn:
            for name, param in self.model.named_parameters():
                if param.requires_grad and 'fc' not in name:
                    param.requires_grad = False

    def forward(self, img):
        # out = self.channel_expander(img)
        batch, channel, height, width = img.shape
        img = img.expand([batch, 3, height, width])
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
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.model(x)
        return self.log_softmax(out)


class JointClassifierV1(torch.nn.Module):
    def __init__(
        self,
        n_class: int,
        total_keypoints: int,
        keypoint_extractor_ckpt: str,
        freeze_cnn: bool = False,
        lambda_cnn: float = 0.8,
    ) -> None:
        super().__init__()
        self.lambda_cnn = lambda_cnn

        self.keypoint_extractor = CustomKeypointExtractor(total_keypoints * 2)
        self.keypoint_extractor.load_checkpoint(keypoint_extractor_ckpt)
        for param in self.keypoint_extractor.parameters():
            param.requires_grad = False
        self.keypoint_extractor.eval()

        self.cnn_model = CNNClassifier(n_class, freeze_cnn)
        self.keypoint_classifier = KeypointClassifier(total_keypoints * 2, n_class)

    def forward(self, img):
        with torch.no_grad():
            keypoints = self.keypoint_extractor(img)
        cnn_out = self.cnn_model(img)
        keypoint_out = self.keypoint_classifier(keypoints)
        return cnn_out * self.lambda_cnn + keypoint_out * (1 - self.lambda_cnn)


class JointClassifierV2(torch.nn.Module):
    def __init__(
        self,
        n_class: int,
        total_keypoints: int,
        keypoint_extractor_ckpt: str,
        freeze_cnn: bool = False,
        lambda_cnn: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_cnn = lambda_cnn

        self.keypoint_extractor = CustomKeypointExtractor(total_keypoints * 2)
        self.keypoint_extractor.load_checkpoint(keypoint_extractor_ckpt)
        for param in self.keypoint_extractor.parameters():
            param.requires_grad = False
        self.keypoint_extractor.eval()

        self.cnn_model = CNNClassifier(n_class, freeze_cnn)
        self.cnn_model.model.fc = torch.nn.Identity()
        self.cnn_model.log_softmax = torch.nn.Identity()
        self.head = torch.nn.Linear(2048 + 2 * total_keypoints, n_class)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, img):
        with torch.no_grad():
            keypoints = self.keypoint_extractor(img)
        cnn_out = self.cnn_model(img)
        out = torch.cat([cnn_out, keypoints], dim=1)
        # return self.log_softmax(self.head(out))
        return self.log_softmax(self.head(out))


if __name__ == '__main__':
    model = CNNClassifier(7)
