import torch

from tsm.ternary import TernaryProject


def test_ternary_forward_and_clipped_ste_backward():
    x = torch.tensor([-2.0, -0.2, 0.0, 0.2, 2.0], requires_grad=True)
    y = TernaryProject.apply(x, torch.tensor(0.15), torch.tensor(2.0))

    assert y.tolist() == [-2.0, -2.0, 0.0, 2.0, 2.0]

    y.sum().backward()
    assert x.grad.tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]
