from torch import nn

from rlrl.modules import evaluating

if __name__ == "__main__":
    net1 = nn.Sequential(
        nn.Linear(10, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )

    net2 = nn.Sequential(
        nn.Linear(10, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )

    net3 = nn.Sequential(
        nn.Linear(10, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )

    with evaluating(net1, net2, net3) as (n1, n2, n3):
        print(n1.training)
        print(n2.training)
        print(n3.training)

    print(net1.training)
    print(net2.training)
    print(net3.training)
