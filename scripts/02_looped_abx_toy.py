"""Minimal looped-LoRA model: f(x) = (w0 + a*b)^K * x, loss = 0.5*(f - s)^2 at x = 1.

K = 1 is an ordinary LoRA layer; K = 3 is a layer applied three times with shared
weights. Checks three closed-form claims numerically:

 1. Gradient amplification at LoRA init: dL/d(ab) is K*w0^(K-1) times the K=1 value
    for the same output error.
 2. Sharpness at a minimum: top Hessian eigenvalue = (K*u^(K-1))^2 * (a^2 + b^2),
    u = s^(1/K); plain GD is stable iff lr < 2/sharpness.
 3. Sign descent (the scale-free core of Adam and Muon) takes the same weight-space
    step for every K, so the first-step change in the output is K*w0^(K-1) times larger.
"""
import torch

torch.set_default_dtype(torch.float64)


def loss(a, b, w0, K, s):
    return 0.5 * ((w0 + a * b) ** K - s) ** 2


w0, a0 = 1.0, 0.5
print("1) gradient wrt b at init (b=0), same output error e = f - s = -0.3")
for K in (1, 3):
    a, b = torch.tensor(a0), torch.tensor(0.0, requires_grad=True)
    s = w0**K + 0.3
    loss(a, b, w0, K, s).backward()
    print(f"   K={K}: grad_b={b.grad.item():+.4f}  predicted={-0.3 * K * w0**(K-1) * a0:+.4f}")

print("2) sharpness at a balanced minimum and the GD stability edge")
for K in (1, 3):
    s = 2.0
    u = s ** (1 / K)
    ab = u - w0
    a_star = b_star = ab**0.5
    H = torch.autograd.functional.hessian(lambda v: loss(v[0], v[1], w0, K, s), torch.tensor([a_star, b_star]))
    lam = torch.linalg.eigvalsh(H).max().item()
    pred = (K * u ** (K - 1)) ** 2 * (a_star**2 + b_star**2)
    out = []
    for mult in (0.95, 1.05):
        lr = mult * 2 / lam
        v = torch.tensor([a_star + 1e-6, b_star - 2e-6])
        d0 = None
        for t in range(400):
            v = v.clone().requires_grad_(True)
            L = loss(v[0], v[1], w0, K, s)
            (g,) = torch.autograd.grad(L, v)
            v = (v - lr * g).detach()
            if t == 0:
                d0 = L.item()
        out.append(f"lr={mult:.2f}*2/lam -> loss x{loss(v[0], v[1], w0, K, s).item() / d0:.1e}")
    print(f"   K={K}: sharpness={lam:.4f} predicted={pred:.4f} | " + " ; ".join(out))

print("3) one sign-descent step from LoRA init, lr = 1e-3")
for K in (1, 3):
    a, b = torch.tensor(a0), torch.tensor(0.0, requires_grad=True)
    s = w0**K + 0.3
    loss(a, b, w0, K, s).backward()
    db = -1e-3 * torch.sign(b.grad)
    df = (w0 + a * (b + db)).item() ** K - w0**K
    print(f"   K={K}: |delta b|={abs(db.item()):.1e}  delta f={df:.3e}  predicted={K * w0**(K-1) * a0 * 1e-3:.3e}")
