import numpy as np


def _gcf_constants(method, p, m):
    if method == 'varimax':
        return (0.0, (p - 1) / p, 1 / p, -1.0)
    if method == 'quartimax':
        return (0.0, 1.0, 1.0, -1.0)
    if method == 'equamax':
        return (0.0, 1 - m / (2.0 * p), m / (2.0 * p), -1.0)
    if method == 'parsimax':
        return (0.0, 1 - (m - 1) / (p + m - 2), (m - 1) / (p + m - 2), -1.0)
    raise ValueError(f"unknown GCF method: {method}")


class GCFCriterion:

    def __init__(self, method, p, m):
        self.method = method
        self.p, self.m = p, m
        self.k1, self.k2, self.k3, self.k4 = _gcf_constants(method, p, m)

    def Q(self, L):
        B = L * L
        r = np.sum(B, axis=1, keepdims=True)
        c = np.sum(B, axis=0, keepdims=True)
        s = np.sum(c)
        return 0.25 * (self.k1 * s * s
                       + self.k2 * np.sum(r * r)
                       + self.k3 * np.sum(c * c)
                       + self.k4 * np.sum(B * B))

    def dQ(self, L):
        B = L * L
        r = np.sum(B, axis=1, keepdims=True)
        c = np.sum(B, axis=0, keepdims=True)
        s = np.sum(c)
        W = self.k1 * s + self.k2 * r + self.k3 * c + self.k4 * B
        return L * W

    def d2Q_apply(self, L, Y):
        p, m = self.p, self.m
        Y2d = Y if Y.ndim == 2 else Y[:, None]
        k = Y2d.shape[1]
        B = L * L
        r = np.sum(B, axis=1, keepdims=True)
        c = np.sum(B, axis=0, keepdims=True)
        s = np.sum(c)
        W = self.k1 * s + self.k2 * r + self.k3 * c + self.k4 * B
        vL = L.reshape(-1, order='F')[:, None]
        vW = W.reshape(-1, order='F')[:, None]
        Y1 = vL * Y2d
        Y3 = Y1.reshape(p, m, k, order='F')
        RY = np.zeros_like(Y1)
        if self.k1 != 0.0:
            RY += self.k1 * Y1.sum(axis=0, keepdims=True)
        if self.k2 != 0.0:
            sp_ = Y3.sum(axis=1)
            RY += self.k2 * np.broadcast_to(sp_[:, None, :], (p, m, k)).reshape(p * m, k, order='F')
        if self.k3 != 0.0:
            sm_ = Y3.sum(axis=0)
            RY += self.k3 * np.broadcast_to(sm_[None, :, :], (p, m, k)).reshape(p * m, k, order='F')
        out = vL * (2.0 * RY + 2.0 * self.k4 * Y1) + vW * Y2d
        return out if Y.ndim == 2 else out[:, 0]

    def d2Q(self, L):
        pm = self.p * self.m
        return self.d2Q_apply(L, np.eye(pm))


class TargetCriterion:

    def __init__(self, H, W=None):
        self.H = np.asarray(H)
        self.W = np.ones_like(self.H) if W is None else np.asarray(W)
        self.p, self.m = self.H.shape

    def Q(self, L):
        return 0.5 * np.sum(self.W * (L - self.H) ** 2)

    def dQ(self, L):
        return self.W * (L - self.H)

    def d2Q_apply(self, L, Y):
        vW = self.W.reshape(-1, order='F')[:, None] if Y.ndim == 2 else self.W.reshape(-1, order='F')
        return vW * Y

    def d2Q(self, L):
        return np.diag(self.W.reshape(-1, order='F'))
