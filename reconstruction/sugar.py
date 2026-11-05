"""SUGAR (Stein Unbiased GrAdient estimator of the Risk) of the MC-SURE used in the paper, and minimization of the
MC-SURE over the hyperparameters mu = (mu_smooth, mu_sparse) with a quasi-Newton method using SUGAR.

MC-SURE (fixed metric W = C_y^{-1}, estimated from the data y, as in Fig. 5):

    SURE(mu) = D(x(y)) - N + 2 div,
    D(x)     = sum_n || centered(P_n (y - A x)) ||^2_W,
    div      = (1/xi) delta^T [ centered(A (x(y + xi delta) - A x(y)) ] + (1/xi) delta^T mean_t(xi delta),

where x(y) = x_hat(y, mu) minimizes F(x; y, mu) = phi(y - A x) + mu_smooth R_smooth(x) + mu_sparse R_sparse(x) over
x >= 0 and centered() removes the per-channel temporal mean (plug-in prediction Theta = A x + mean_t(y - A x)).

SUGAR. At the minimizer, grad_x F(x_hat; mu) = 0 on the free variables (x_hat > 0), hence (implicit function theorem)
d x_hat / d mu_i = - H^{-1} grad R_i(x_hat), H the Hessian of F restricted to the free variables. With
g_delta = (2/xi) A^T centered(delta) (gradient of the linear part of 2 div):

    d SURE / d mu_i = (grad D(x_hat) - g_delta)^T d x_hat/d mu_i + g_delta^T d x_hat_eps/d mu_i
                    = - u^T grad R_i(x_hat) - w^T grad R_i(x_hat_eps),
    u = H^{-1} (grad D(x_hat) - g_delta),   w = H_eps^{-1} g_delta    (adjoint method: 2 linear solves per point),

H_eps being the Hessian at x_hat_eps = x_hat(y + xi delta). The linear systems are solved by conjugate gradient, with
Hessian-vector products computed by central finite differences of the gradient of F (the gradient minimized by
run_bfgs, MDREX.objective). The minimization over log(mu) uses L-BFGS-B (scipy) with this gradient.
"""
import numpy as np


def conjugate_gradient(hvp, b, free, maxiter=30, rtol=1e-3, verbose=False, precond=None, snapshots=()):
    """Solve H u = b on the free variables (u = 0 elsewhere) by (preconditioned) conjugate gradient. hvp(v) -> H v
    (full arrays); precond: diagonal d > 0 of a Jacobi preconditioner M = diag(d) (None: no preconditioning).
    Stops after maxiter iterations, at relative residual rtol (||H u - b|| / ||b||), or on a direction of non-positive
    curvature. snapshots: iteration numbers at which (it, u, residual) is stored, returned as a 4th output if not empty
    (several numbers of iterations from a single run)."""
    b = np.where(free, b, 0.0)
    minv = np.where(free, 1.0 / precond, 0.0) if precond is not None else np.where(free, 1.0, 0.0)
    u = np.zeros_like(b)
    r = b.copy()
    z = minv * r
    p = z.copy()
    rz = float(np.dot(r.ravel(), z.ravel()))
    b_norm = np.linalg.norm(b)
    snaps, it, res = [], 0, 1.0
    if b_norm == 0:
        return (u, 0, 0.0, snaps) if snapshots else (u, 0, 0.0)
    for it in range(1, maxiter + 1):
        Hp = np.where(free, hvp(p), 0.0)
        pHp = float(np.dot(p.ravel(), Hp.ravel()))
        if pHp <= 0:
            if verbose:
                print(f"[CG] non-positive curvature at iteration {it}, stop", flush=True)
            it -= 1
            break
        alpha = rz / pHp
        u += alpha * p
        r -= alpha * Hp
        res = np.linalg.norm(r) / b_norm
        if verbose:
            print(f"[CG] it {it}: relative residual {res:.3e}", flush=True)
        if it in snapshots:
            snaps.append((it, u.copy(), res))
        if res <= rtol:
            break
        z = minv * r
        rz_new = float(np.dot(r.ravel(), z.ravel()))
        p = z + (rz_new / rz) * p
        rz = rz_new
    return (u, it, res, snaps) if snapshots else (u, it, res)


def jacobi_preconditioner(hvp, reg_hess, free, n_probes=4, seed=0):
    """Diagonal preconditioner d = diag(H_reg) + c on the free variables. reg_hess = (diagonal, Hessian-vector product)
    of the Hessian H_reg of the regularization (mu_smooth R_smooth + mu_sparse R_sparse), both exact; c = mean diagonal
    of the data part H - H_reg, estimated by Hutchinson (Rademacher probes z on the free variables):
    c = mean_k z_k^T (H z_k - H_reg z_k) / n_free. Returns d and c."""
    reg_diag, reg_hvp = reg_hess
    rng = np.random.default_rng(seed)
    n = int(free.sum())
    tr = 0.0
    for _ in range(n_probes):
        z = np.where(free, rng.choice([-1.0, 1.0], size=free.shape), 0.0)
        tr += float(np.dot(z.ravel(), (hvp(z) - np.where(free, reg_hvp(z), 0.0)).ravel())) / n_probes
    c = max(tr / n, 1e-12 * np.abs(reg_diag[free]).max())
    return np.where(free, reg_diag + c, 1.0), c


def hessian_vector_product(grad_F, x, v, rel_step=1e-3):
    """Central finite difference H v ~ (grad_F(x + h v) - grad_F(x - h v)) / (2 h), h relative to the size of x."""
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        return np.zeros_like(v)
    h = rel_step * max(np.linalg.norm(x), 1e-30) / v_norm
    return (grad_F(x + h * v).astype(np.float64) - grad_F(x - h * v).astype(np.float64)) / (2 * h)


def sugar_gradient(grad_F, grad_F_eps, x_hat, x_hat_eps, grad_D, g_delta, reg_grads, free_tol=0.0,
                   cg_maxiter=30, cg_rtol=1e-3, verbose=False, hvp_step=1e-3, split=False, reg_hess=None,
                   snapshots=()):
    """d SURE / d mu_i, i = (smooth, sparse). grad_F(x) / grad_F_eps(x): gradients of F at y and y + xi delta (current
    mu); reg_grads(x) -> (grad R_smooth(x), grad R_sparse(x)); free variables: x > free_tol * max(x).
    split=True: derivative of 2 div computed as the difference of two solves with the same right-hand side g_delta,
        d (2 div) / d mu_i = (H^{-1} g_delta)^T grad R_i(x_hat) - (H_eps^{-1} g_delta)^T grad R_i(x_hat_eps),
    and derivative of D from its own solve, d D / d mu_i = - (H^{-1} grad D)^T grad R_i(x_hat) (3 linear solves). The two
    terms of the derivative of 2 div scale as 1/xi and nearly cancel: with the same right-hand side and the same number
    of CG iterations (cg_rtol = 0), the errors of the two solves are correlated and cancel too, whereas with the joint
    right-hand side grad D - g_delta of u (split=False) they are independent. The two terms are returned in info.
    reg_hess(x) -> (diagonal, Hessian-vector product) of the Hessian of mu_smooth R_smooth + mu_sparse R_sparse at x
    (edge_preserving_hess): Jacobi preconditioner (jacobi_preconditioner) of the CG solves (None: no preconditioning).
    snapshots: numbers of CG iterations at which the gradient is also computed (same runs of CG); then returns
    [(k, grad, info), ...] for these k instead of (grad, info)."""
    free = x_hat > free_tol * x_hat.max()
    free_eps = x_hat_eps > free_tol * x_hat_eps.max()
    hvp = lambda v: hessian_vector_product(grad_F, x_hat, v, hvp_step)
    hvp_eps = lambda v: hessian_vector_product(grad_F_eps, x_hat_eps, v, hvp_step)
    R = [np.where(free, r, 0).ravel() for r in reg_grads(x_hat)]
    R_eps = [np.where(free_eps, r, 0).ravel() for r in reg_grads(x_hat_eps)]
    pinfo = {}
    if reg_hess is not None:
        M, c = jacobi_preconditioner(hvp, reg_hess(x_hat), free)
        M_eps, c_eps = jacobi_preconditioner(hvp_eps, reg_hess(x_hat_eps), free_eps)
        d = M[free]
        pinfo = dict(precond_c=c, precond_c_eps=c_eps, precond_ratio=float(d.max() / d.min()))
        if verbose:
            print(f"[precond] data constant {c:.3e} / {c_eps:.3e}, max/min of the diagonal {pinfo['precond_ratio']:.3e}",
                  flush=True)
    else:
        M = M_eps = None
    ks = sorted(set(int(k) for k in snapshots if k <= cg_maxiter))
    maxit = max(ks) if ks else cg_maxiter
    snap = tuple(ks) if ks else (maxit,)

    def solve(op, b, fr, P):
        u, it, res, s = conjugate_gradient(op, b, fr, maxit, cg_rtol, verbose, P, snap)
        if not s or s[-1][0] < it:   # converged (rtol) or stopped before the last snapshot
            s.append((it, u, res))
        return {k: next(((uu, rr) for (kk, uu, rr) in s if kk >= k), (u, res)) for k in snap}

    W = solve(hvp_eps, g_delta, free_eps, M_eps)
    if not split:
        U = solve(hvp, grad_D - g_delta, free, M)
    else:
        U = solve(hvp, grad_D, free, M)
        W0 = solve(hvp, g_delta, free, M)
    out = []
    for k in snap:
        (u, res_u), (w, res_w) = U[k], W[k]
        if not split:
            grad = np.array([-np.dot(u.ravel(), R[i]) - np.dot(w.ravel(), R_eps[i]) for i in range(2)])
            info = dict(cg_res_u=res_u, cg_res_w=res_w)
        else:
            w0, res_w0 = W0[k]
            a = np.array([np.dot(w0.ravel(), R[i]) for i in range(2)])       # 1/xi terms of d (2 div) / d mu
            b = np.array([np.dot(w.ravel(), R_eps[i]) for i in range(2)])
            dD = np.array([-np.dot(u.ravel(), R[i]) for i in range(2)])
            grad = dD + a - b
            info = dict(cg_res_u=res_u, cg_res_w=res_w, cg_res_w0=res_w0, dD=dD, d2div=a - b, d2div_a=a, d2div_b=b)
        info.update(pinfo, n_free=int(free.sum()), n_free_eps=int(free_eps.sum()))
        out.append((k, grad, info))
    return out if ks else out[0][1:]


def edge_preserving_hess(x, mu_smooth, R_smooth, stride=2):
    """Hessian of mu_smooth R_smooth at x (torch function of a (C, H, W) tensor), exact by double backward: returns its
    diagonal and its Hessian-vector product v -> H_reg v (numpy). Diagonal by probing: the Hessian of the
    edge-preserving term couples only pixels at distance <= 1 in each direction (and the channels of a pixel), so the
    probes 1 on the pixels (i, j) = (a, b) mod stride of one channel (stride >= 2) recover it with stride^2 C products.
    R_sparse = sum |x| has a zero Hessian on x > 0."""
    import torch
    xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    (g,) = torch.autograd.grad(mu_smooth * R_smooth(xt), xt, create_graph=True)

    def hvp(v):
        (hv,) = torch.autograd.grad(g, xt, grad_outputs=torch.as_tensor(v, dtype=torch.float64), retain_graph=True)
        return hv.detach().numpy()
    C, H, W = x.shape
    diag = np.zeros(x.shape, dtype=np.float64)
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    for ch in range(C):
        for a in range(stride):
            for b in range(stride):
                mask = (ii % stride == a) & (jj % stride == b)
                v = np.zeros(x.shape)
                v[ch][mask] = 1.0
                diag[ch][mask] = hvp(v)[ch][mask]
    return diag, hvp


def hvp_symmetry_test(grad_F, x, free, rel_steps=(1e-4, 1e-3, 1e-2, 1e-1), seed=0):
    """Quality of the finite-difference Hessian-vector products: for random v, w on the free variables,
    |v^T H w - w^T H v| / max(|v^T H w|, |w^T H v|) should be << 1 (H symmetric); also returns v^T H v > 0 ?"""
    rng = np.random.default_rng(seed)
    v = np.where(free, rng.standard_normal(x.shape), 0.0)
    w = np.where(free, rng.standard_normal(x.shape), 0.0)
    out = []
    for h in rel_steps:
        Hv = hessian_vector_product(grad_F, x, v, h)
        Hw = hessian_vector_product(grad_F, x, w, h)
        a, b = float(np.dot(v.ravel(), Hw.ravel())), float(np.dot(w.ravel(), Hv.ravel()))
        out.append(dict(rel_step=h, asym=abs(a - b) / max(abs(a), abs(b), 1e-300), vHv=float(np.dot(v.ravel(), Hv.ravel()))))
    return out
