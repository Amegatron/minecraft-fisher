import torch.nn


class FishermanModel(torch.nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features

        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 6, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 128, 4, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(128, 256, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Dropout2d(0.5),
            torch.nn.Flatten(),
            torch.nn.Linear(8192, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(0.5),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, out_features),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.network(x)
