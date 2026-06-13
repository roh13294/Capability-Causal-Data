from causal_reliability.models.cnn import SmallCNN
from causal_reliability.models.mlp import MLP
from causal_reliability.models.tabular_mlp import TabularMLP
from causal_reliability.models.tiny_transformer import TinyTransformerClassifier

__all__ = ["MLP", "SmallCNN", "TabularMLP", "TinyTransformerClassifier", "build_model"]


def build_model(task_type: str, input_shape: tuple[int, ...], num_classes: int = 2):
    if task_type == "vision":
        return SmallCNN(num_classes=num_classes)
    if task_type == "text":
        return TinyTransformerClassifier(num_classes=num_classes)
    if task_type == "tabular":
        return TabularMLP(input_dim=input_shape[0], num_classes=num_classes)
    return MLP(input_dim=input_shape[0], num_classes=num_classes)
