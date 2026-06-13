import torch


class SmallCNN(torch.nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 12, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(12, 24, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.head = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(24 * 2 * 2, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x.float()))
