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
    def __init__(
        self,
        n_class: int,
        total_keypoints: int,
        keypoint_extractor_ckpt: str,
    ):
        super().__init__()
        self.keypoint_extractor = CustomKeypointExtractor(total_keypoints * 2)
        self.keypoint_extractor.load_checkpoint(keypoint_extractor_ckpt)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(total_keypoints * 2, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, n_class),
        )
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, img):
        with torch.no_grad():
            keypoints = self.keypoint_extractor(img)
        out = self.model(keypoints)
        return self.log_softmax(out)


class KeypointClassifierMediapipe(torch.nn.Module):
    def __init__(
        self,
        n_class: int,
        total_keypoints: int,
    ):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(total_keypoints * 3, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, n_class),
        )
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, img, keypoints):
        out = self.model(keypoints)
        return self.log_softmax(out)


class JointClassifier(torch.nn.Module):
    def __init__(
        self,
        n_class: int,
        total_keypoints: int,
        keypoint_extractor_ckpt: str,
        freeze_cnn: bool = False,
    ) -> None:
        super().__init__()

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

    def forward(self, img, keypoints):
        with torch.no_grad():
            keypoints = self.keypoint_extractor(img)
        cnn_out = self.cnn_model(img)
        out = torch.cat([cnn_out, keypoints], dim=1)
        return self.log_softmax(self.head(out))


class JointClassifierMediapipe(torch.nn.Module):
    def __init__(
        self,
        n_class: int,
        total_keypoints: int,
        freeze_cnn: bool = False,
    ) -> None:
        super().__init__()
        self.cnn_model = CNNClassifier(n_class, freeze_cnn)
        self.cnn_model.model.fc = torch.nn.Identity()
        self.cnn_model.log_softmax = torch.nn.Identity()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(2048 + 3 * total_keypoints, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, n_class),
        )
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, img, keypoints):
        cnn_out = self.cnn_model(img)
        out = torch.cat([cnn_out, keypoints], dim=1)
        return self.log_softmax(self.head(out))


if __name__ == '__main__':
    model = CNNClassifier(7)
