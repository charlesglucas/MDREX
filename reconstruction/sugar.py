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
                   cg_maxiter=30, cg_rtol=1e-3, verbose=False):
    """d SURE / d mu_i, i = (smooth, sparse). grad_F(x) / grad_F_eps(x): gradients of F at y and y + xi delta (current
    mu); reg_grads(x) -> (grad R_smooth(x), grad R_sparse(x)); free variables: x > free_tol * max(x)."""
    free = x_hat > free_tol * x_hat.max()
    free_eps = x_hat_eps > free_tol * x_hat_eps.max()
    u, it_u, res_u = conjugate_gradient(lambda v: hessian_vector_product(grad_F, x_hat, v), grad_D - g_delta, free,
                                        cg_maxiter, cg_rtol, verbose)
    w, it_w, res_w = conjugate_gradient(lambda v: hessian_vector_product(grad_F_eps, x_hat_eps, v), g_delta, free_eps,
                                        cg_maxiter, cg_rtol, verbose)
    R = reg_grads(x_hat)
    R_eps = reg_grads(x_hat_eps)
    grad = np.array([-np.dot(u.ravel(), np.where(free, R[i], 0).ravel())
                     - np.dot(w.ravel(), np.where(free_eps, R_eps[i], 0).ravel()) for i in range(2)])
    info = dict(cg_iter_u=it_u, cg_res_u=res_u, cg_iter_w=it_w, cg_res_w=res_w,
                n_free=int(free.sum()), n_free_eps=int(free_eps.sum()))
    return grad, info
