"""Role-independent reproducer for PyTorch issue #99638."""

from __future__ import annotations

import json

import torch


def packed_masked(x, sizes):
    batch, time, *rest = x.shape
    mask = torch.arange(time, device=x.device)[None, :] < sizes[:, None]
    mask = mask.view(batch, time, *[1] * len(rest))
    return torch.masked_select(x, mask).view(-1, *rest)


def packed_indexed(x, sizes):
    _, _, *rest = x.shape
    mask = torch.arange(x.shape[1], device=x.device)[None, :] < sizes[:, None]
    index = mask.reshape(-1).nonzero().flatten()
    return torch.index_select(x.reshape(-1, *rest), 0, index)


def gradient(pack):
    torch.manual_seed(42)
    net = torch.nn.Conv1d(5, 5, 5, padding="same")
    inputs = torch.randn(3, 4, 5, requires_grad=True)
    targets = torch.randint(0, 5, (3, 4))
    sizes = torch.tensor([4, 3, 2], dtype=torch.int32)
    logits = net(inputs.transpose(1, 2)).transpose(1, 2)
    loss = torch.nn.CrossEntropyLoss(reduction="sum")(
        pack(logits, sizes), pack(targets, sizes).long()
    )
    return torch.autograd.grad(loss, net.bias)[0].detach()


def main():
    observed = gradient(packed_masked)
    reference = gradient(packed_indexed)
    passed = bool(torch.allclose(observed, reference, rtol=1e-10, atol=1e-10))
    print(json.dumps({
        "schema": "P4-Role-Independent-Runner-v1",
        "case_id": "pytorch_99638",
        "package": "torch",
        "version": torch.__version__,
        "outcome": "pass" if passed else "fail",
        "invocations": 2,
        "details": {
            "max_absolute_gradient_error": float((observed - reference).abs().max()),
            "observed": observed.tolist(),
            "reference": reference.tolist(),
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
