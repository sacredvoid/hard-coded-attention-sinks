"""Linear probe on frozen features: standardized multinomial logistic regression (L-BFGS)."""

import torch
import torch.nn.functional as F


def _fit(x: torch.Tensor, y: torch.Tensor, n_cls: int, wd: float) -> torch.Tensor:
    w = torch.zeros(x.shape[1] + 1, n_cls, device=x.device, requires_grad=True)
    xb = torch.cat([x, torch.ones(len(x), 1, device=x.device)], 1)
    opt = torch.optim.LBFGS([w], lr=1, max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(xb @ w, y) + wd * w[:-1].pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach()


def _acc(w, x, y) -> float:
    xb = torch.cat([x, torch.ones(len(x), 1, device=x.device)], 1)
    return ((xb @ w).argmax(1) == y).float().mean().item()


def linear_probe(xtr, ytr, xte, yte, wds=(1e-4, 1e-3, 1e-2)) -> float:
    """Pick weight decay on a 20% split of train, refit on all train, return test accuracy."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xtr, xte = xtr.float().to(dev), xte.float().to(dev)
    ytr, yte = ytr.long().to(dev), yte.long().to(dev)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    n_cls = int(max(ytr.max(), yte.max())) + 1
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(xtr), generator=g).to(dev)
    cut = int(0.8 * len(xtr))
    a, b = perm[:cut], perm[cut:]
    best = max(wds, key=lambda wd: _acc(_fit(xtr[a], ytr[a], n_cls, wd), xtr[b], ytr[b]))
    return _acc(_fit(xtr, ytr, n_cls, best), xte, yte)


if __name__ == "__main__":  # separable toy: probe must be ~perfect; noise: ~chance
    torch.manual_seed(0)
    y = torch.randint(0, 3, (600,))
    x = torch.randn(600, 8) + 4 * F.one_hot(y, 8).float()
    assert linear_probe(x[:400], y[:400], x[400:], y[400:]) > 0.95
    assert linear_probe(torch.randn(400, 8), y[:400], torch.randn(200, 8), y[400:]) < 0.55
    print("probe selftest ok")
