import torch

from rlrl.utils import conjugate_gradient

if __name__ == "__main__":
    _A = torch.rand(3, 3)
    A = _A @ torch.t(_A)
    x_true = torch.rand(3)
    b = torch.matmul(A, x_true)

    x_cg = conjugate_gradient(lambda x: torch.matmul(A, x), b)

    print(x_true)
    print(x_cg)
