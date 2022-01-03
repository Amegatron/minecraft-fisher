import torch.nn


class FishermanModel(torch.nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features

        self.alex_net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 96, 11, 4),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2),
            torch.nn.Conv2d(96, 256, 5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2),
            torch.nn.Conv2d(256, 384, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(384, 384, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(384, 256, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2),
            torch.nn.Dropout2d(0.5),
            torch.nn.Flatten(),
            torch.nn.Linear(9216, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, out_features),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.alex_net(x)
