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


def conjugate_gradient(hvp, b, free, maxiter=30, rtol=1e-3, verbose=False):
    """Solve H u = b on the free variables (u = 0 elsewhere) by conjugate gradient. hvp(v) -> H v (full arrays).
    Stops after maxiter iterations, at relative residual rtol, or on a direction of non-positive curvature."""
    b = np.where(free, b, 0.0)
    u = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rr = float(np.dot(r.ravel(), r.ravel()))
    b_norm = np.sqrt(rr)
    if b_norm == 0:
        return u, 0, 0.0
    for it in range(1, maxiter + 1):
        Hp = np.where(free, hvp(p), 0.0)
        pHp = float(np.dot(p.ravel(), Hp.ravel()))
        if pHp <= 0:
            if verbose:
                print(f"[CG] non-positive curvature at iteration {it}, stop", flush=True)
            break
        alpha = rr / pHp
        u += alpha * p
        r -= alpha * Hp
        rr_new = float(np.dot(r.ravel(), r.ravel()))
        if verbose:
            print(f"[CG] it {it}: relative residual {np.sqrt(rr_new) / b_norm:.3e}", flush=True)
        if np.sqrt(rr_new) <= rtol * b_norm:
            rr = rr_new
            break
        p = r + (rr_new / rr) * p
        rr = rr_new
    return u, it, np.sqrt(rr) / b_norm


def hessian_vector_product(grad_F, x, v, rel_step=1e-3):
    """Central finite difference H v ~ (grad_F(x + h v) - grad_F(x - h v)) / (2 h), h relative to the size of x."""
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        return np.zeros_like(v)
    h = rel_step * max(np.linalg.norm(x), 1e-30) / v_norm
    return (grad_F(x + h * v).astype(np.float64) - grad_F(x - h * v).astype(np.float64)) / (2 * h)


def sugar_gradient(grad_F, grad_F_eps, x_hat, x_hat_eps, grad_D, g_delta, reg_grads, free_tol=0.0,
                   cg_maxiter=30, cg_rtol=1e-3, verbose=False, hvp_step=1e-3, split=False):
    """d SURE / d mu_i, i = (smooth, sparse). grad_F(x) / grad_F_eps(x): gradients of F at y and y + xi delta (current
    mu); reg_grads(x) -> (grad R_smooth(x), grad R_sparse(x)); free variables: x > free_tol * max(x).
    split=True: derivative of 2 div computed as the difference of two solves with the same right-hand side g_delta,
        d (2 div) / d mu_i = (H^{-1} g_delta)^T grad R_i(x_hat) - (H_eps^{-1} g_delta)^T grad R_i(x_hat_eps),
    and derivative of D from its own solve, d D / d mu_i = - (H^{-1} grad D)^T grad R_i(x_hat) (3 linear solves). The two
    terms of the derivative of 2 div scale as 1/xi and nearly cancel: with the same right-hand side and the same number
    of CG iterations (cg_rtol = 0), the errors of the two solves are correlated and cancel too, whereas with the joint
    right-hand side grad D - g_delta of u (split=False) they are independent. The two terms are returned in info."""
    free = x_hat > free_tol * x_hat.max()
    free_eps = x_hat_eps > free_tol * x_hat_eps.max()
    hvp = lambda v: hessian_vector_product(grad_F, x_hat, v, hvp_step)
    hvp_eps = lambda v: hessian_vector_product(grad_F_eps, x_hat_eps, v, hvp_step)
    R = [np.where(free, r, 0).ravel() for r in reg_grads(x_hat)]
    R_eps = [np.where(free_eps, r, 0).ravel() for r in reg_grads(x_hat_eps)]
    w, it_w, res_w = conjugate_gradient(hvp_eps, g_delta, free_eps, cg_maxiter, cg_rtol, verbose)
    if not split:
        u, it_u, res_u = conjugate_gradient(hvp, grad_D - g_delta, free, cg_maxiter, cg_rtol, verbose)
        grad = np.array([-np.dot(u.ravel(), R[i]) - np.dot(w.ravel(), R_eps[i]) for i in range(2)])
        info = dict(cg_iter_u=it_u, cg_res_u=res_u, cg_iter_w=it_w, cg_res_w=res_w)
    else:
        u, it_u, res_u = conjugate_gradient(hvp, grad_D, free, cg_maxiter, cg_rtol, verbose)
        w0, it_w0, res_w0 = conjugate_gradient(hvp, g_delta, free, cg_maxiter, cg_rtol, verbose)
        a = np.array([np.dot(w0.ravel(), R[i]) for i in range(2)])       # 1/xi terms of d (2 div) / d mu
        b = np.array([np.dot(w.ravel(), R_eps[i]) for i in range(2)])
        dD = np.array([-np.dot(u.ravel(), R[i]) for i in range(2)])
        grad = dD + a - b
        info = dict(cg_iter_u=it_u, cg_res_u=res_u, cg_iter_w=it_w, cg_res_w=res_w, cg_iter_w0=it_w0, cg_res_w0=res_w0,
                    dD=dD, d2div=a - b, d2div_a=a, d2div_b=b)
    info.update(n_free=int(free.sum()), n_free_eps=int(free_eps.sum()))
    return grad, info


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
