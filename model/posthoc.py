import torch
import torch.nn as nn
import torch.nn.functional as F

from train.utils import to_probs


def _safe_entropy(probs: torch.Tensor) -> torch.Tensor:
  # Softmax can underflow to exact zeros for very confident logits.
  safe_probs = probs.clamp_min(1e-12)
  return -(safe_probs * safe_probs.log()).sum(dim=-1).unsqueeze(-1)


def from_str_to_posthoc_cls(posthoc_str: str) -> type[nn.Module]:
  match posthoc_str:
    case "standard":
      posthoc = PostHocMLP
    case "large":
      posthoc = PostHocLargeMLP
    case "small":
      posthoc = PostHocSmallMLP
    case _:
      raise ValueError(f"Unknown posthoc str: {posthoc_str}")

  return posthoc


class PostHocMLP(nn.Module):
  def __init__(
    self,
    num_classes: int,
    top_k: int = 10,
    temp: float = 1.0,
  ) -> None:
    super().__init__()

    self.num_classes = num_classes
    self.top_k = min(top_k, num_classes)
    self.temp = temp

    self.relu = nn.ReLU(inplace=True)
    self.mlp1 = nn.Linear(1 + num_classes + self.top_k, 2**6)
    self.mlp2 = nn.Linear(2**6, 2**4)
    self.mlp3 = nn.Linear(2**4, 1)

  def input_transform(self, x: torch.Tensor, small_out: torch.Tensor) -> torch.Tensor:
    del x

    sm_logits = small_out
    sm_probs = to_probs(sm_logits, self.temp, dim=1)

    # Appendix C.2 uses entropy, top-k probabilities, and the argmax one-hot.
    entropy = _safe_entropy(sm_probs)
    top_probs = torch.topk(sm_probs, k=self.top_k, dim=-1).values
    one_hot = F.one_hot(sm_probs.argmax(dim=-1), num_classes=self.num_classes)

    return torch.hstack([entropy, top_probs, one_hot])

  def forward(self, x_and_small_out: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    x = self.input_transform(*x_and_small_out)

    x = self.relu(self.mlp1(x))
    x = self.relu(self.mlp2(x))
    x = self.mlp3(x)

    return x.squeeze()  # We want this to output a vector


class PostHocLargeMLP(nn.Module):
  def __init__(
    self,
    num_classes: int,
    top_k: int = 10,
    temp: float = 1.0,
  ) -> None:
    super().__init__()

    self.num_classes = num_classes
    self.top_k = min(top_k, num_classes)
    self.temp = temp

    self.relu = nn.ReLU(inplace=True)
    self.mlp1 = nn.Linear(1 + num_classes + self.top_k, 2**6)
    self.mlp2 = nn.Linear(2**6, 2**4)
    self.mlp3 = nn.Linear(2**4, 2**2)
    self.mlp4 = nn.Linear(2**2, 1)

  def input_transform(self, x: torch.Tensor, small_out: torch.Tensor) -> torch.Tensor:
    del x

    sm_logits = small_out
    sm_probs = to_probs(sm_logits, self.temp, dim=1)

    entropy = _safe_entropy(sm_probs)
    top_probs = torch.topk(sm_probs, k=self.top_k, dim=-1).values
    one_hot = F.one_hot(sm_probs.argmax(dim=-1), num_classes=self.num_classes)

    return torch.hstack([entropy, top_probs, one_hot])

  def forward(self, x_and_small_out: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    # _, x = x_and_small_out  # Only use the small output
    x = self.input_transform(*x_and_small_out)

    x = self.relu(self.mlp1(x))
    x = self.relu(self.mlp2(x))
    x = self.relu(self.mlp3(x))
    x = self.mlp4(x)

    return x.squeeze()  # We want this to output a vector


class PostHocSmallMLP(nn.Module):
  def __init__(
    self,
    num_classes: int,
    top_k: int = 10,
    temp: float = 1.0,
  ) -> None:
    super().__init__()

    self.num_classes = num_classes
    self.top_k = min(top_k, num_classes)
    self.temp = temp

    self.relu = nn.ReLU(inplace=True)
    self.mlp1 = nn.Linear(1 + num_classes + self.top_k, 2**6)
    self.mlp2 = nn.Linear(2**6, 1)

  def input_transform(self, x: torch.Tensor, small_out: torch.Tensor) -> torch.Tensor:
    del x

    sm_logits = small_out
    sm_probs = to_probs(sm_logits, self.temp, dim=1)

    entropy = _safe_entropy(sm_probs)
    top_probs = torch.topk(sm_probs, k=self.top_k, dim=-1).values
    one_hot = F.one_hot(sm_probs.argmax(dim=-1), num_classes=self.num_classes)

    return torch.hstack([entropy, top_probs, one_hot])

  def forward(self, x_and_small_out: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    x = self.input_transform(*x_and_small_out)

    x = self.relu(self.mlp1(x))
    x = self.mlp2(x)

    return x.squeeze()  # We want this to output a vector
