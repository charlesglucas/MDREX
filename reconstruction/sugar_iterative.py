"""MC-SUGAR by iterative (forward-mode) differentiation of the reconstruction algorithm (Deledalle et al. 2014), instead
of the implicit differentiation of reconstruction/sugar.py.

Why: the reconstructions stop on the variables-change test of VMLMB with a projected gradient still ~1/10 of its initial
norm, so that x_hat is not a stationary point of F and the implicit formula d x_hat / d mu = - H^{-1} grad R (which
assumes grad F(x_hat) = 0 on the free variables) differentiates a minimizer that the algorithm does not reach; it also
needs ~640 CG iterations per solve (ill-conditioned data term). Here the derivative of the actual output of the
algorithm is propagated along its iterations, as the Monte Carlo divergence of MC-SURE is computed on these outputs.

vmlmb_diff is a copy of utils/optm.vmlmb (same iterations: projection on the bounds, L-BFGS two-loop recursion on the
free variables, Armijo line search with parabolic backtracking) restricted to what run_bfgs uses (lower bound only,
mem = 5, no BLMVM, epsilon = 0), which also propagates the tangents X_j = d x_k / d mu_j, j = 1..n_dir:
  - trial point x = max(x0 + alpha d, lower): X = 1[x0 + alpha d > lower] (X0 + alpha D), the step alpha (line search
    and step limit) and the active set being held fixed (piecewise derivative, standard for iterative differentiation);
  - gradient at an iterate: G_j = d g / d mu_j = H(x) X_j + grad R_j(x) (g = grad F, F linear in mu), with H(x) X_j by a
    finite difference of the gradient of F along X_j (one-sided: g(x) is known, 1 extra gradient per direction and per
    iteration; central: 2);
  - L-BFGS memory: tangents of s = x - x0 and y = g - g0; search direction: tangent of the two-loop recursion (product
    rule on every inner product), the set of free variables and the acceptance tests being held fixed.
The utils/optm.py code is not modified.

MC-SURE(mu) = D(x_hat) - N + 2 div, 2 div = g_delta^T (x_hat_eps - x_hat) + const (g_delta = (2/xi) A^T centered(delta)):
    d SURE / d mu_j = grad D(x_hat)^T X_j + g_delta^T (X_eps_j - X_j),
X and X_eps the tangents of the two reconstructions (y and y + xi delta), both computed by vmlmb_diff.
"""
import sys

import numpy as np

import utils.optm as optm


def _dot(a, b):
    return float(np.vdot(a, b))


class LBFGSDiff:
    """L-BFGS memory of utils/optm.LBFGS (masked two-loop recursion of LBFGS.apply with freevars) with the tangents of the
    stored pairs (s, y) along n_dir directions."""

    def __init__(self, m):
        self.m, self.mp, self.itr = m, 0, 0
        self.S, self.Y = [None] * m, [None] * m
        self.dS, self.dY = [None] * m, [None] * m

    def update(self, s, y, ds, dy):
        if _dot(s, y) <= 0:
            return False
        k = self.itr % self.m
        self.S[k], self.Y[k], self.dS[k], self.dY[k] = s, y, ds, dy
        self.itr += 1
        self.mp = min(self.mp + 1, self.m)
        return True

    def apply(self, q, dq, msk):
        """d = B q on the free variables (msk) and its tangent dd (n_dir, ...) given dq (n_dir, ...); returns
        (d, dd, scaled) as LBFGS.apply(q, freevars)."""
        m, mp = self.m, self.mp
        nd = dq.shape[0]
        last = self.itr + m
        inds = [i % m for i in range(last - mp, last)]
        d = msk * q
        dd = msk * dq
        gamma, dgamma = 0.0, np.zeros(nd)
        alpha, dalpha, rho, drho = {}, {}, {}, {}
        for i in reversed(inds):
            s_i, y_i = self.S[i], msk * self.Y[i]
            ds_i, dy_i = self.dS[i], msk * self.dY[i]
            rho_i = _dot(s_i, y_i)
            drho_i = np.array([_dot(ds_i[j], y_i) + _dot(s_i, dy_i[j]) for j in range(nd)])
            if rho_i > 0:
                if gamma <= 0.0:
                    yy = _dot(y_i, y_i)
                    gamma = rho_i / yy
                    dgamma = np.array([drho_i[j] / yy - rho_i * 2 * _dot(y_i, dy_i[j]) / yy ** 2 for j in range(nd)])
                a = _dot(d, s_i) / rho_i
                da = np.array([(_dot(dd[j], s_i) + _dot(d, ds_i[j])) / rho_i - a * drho_i[j] / rho_i for j in range(nd)])
                dd = dd - da[:, None, None, None] * y_i[None] - a * dy_i
                d = d - a * y_i
                alpha[i], dalpha[i] = a, da
            rho[i], drho[i] = rho_i, drho_i
        if gamma > 0 and gamma != 1:
            dd = gamma * dd + dgamma[:, None, None, None] * d[None]
            d = gamma * d
        for i in inds:
            if rho[i] > 0:
                Y_i, dY_i = self.Y[i], self.dY[i]
                b = _dot(d, Y_i) / rho[i]
                db = np.array([(_dot(dd[j], Y_i) + _dot(d, dY_i[j])) / rho[i] - b * drho[i][j] / rho[i] for j in range(nd)])
                S_i, dS_i = msk * self.S[i], msk * self.dS[i]
                dd = dd + (dalpha[i] - db)[:, None, None, None] * S_i[None] + (alpha[i] - b) * dS_i
                d = d + (alpha[i] - b) * S_i
        return d, dd, gamma > 0


def vmlmb_diff(fg, x0, dgrad, lower=0.0, mem=5, ftol=1e-8, gtol=1e-5, xtol=1e-6, maxiter=np.inf, maxeval=np.inf,
               hvp_step=1e-2, central=False, freeze_metric=True, verb=0, output=sys.stdout):
    """VMLMB of utils/optm.vmlmb (lower bound, mem, ftol, gtol, xtol as in run_bfgs) with forward-mode differentiation.
    fg(x) -> (f, g); dgrad(x) -> array (n_dir, ...) of the partial derivatives d g / d mu_j at fixed x (grad R_j(x)).
    Returns (x, f, g, status, X) with X (n_dir, ...) the tangents d x / d mu_j of the returned x, and info (number of
    evaluations of fg for the reconstruction and for the tangents).
    freeze_metric=True: the L-BFGS metric B_k is held fixed in the derivative (tangents of the memory pairs not
    propagated): X <- 1[active] (X - alpha B_k (H X + grad R)), a Richardson iteration on H X = - grad R preconditioned by
    the metric of VMLMB. freeze_metric=False: full derivative of the two-loop recursion (unstable: the divisions by
    s^T y amplify the tangents)."""
    Inf = np.inf
    fatol, frtol = optm.get_tolerances(ftol, atol=-Inf)
    gatol, grtol = optm.get_tolerances(gtol, atol=0.0)
    xatol, xrtol = optm.get_tolerances(xtol, atol=0.0)
    lnsrch = optm.LineSearch()
    x = np.asarray(x0, dtype=np.float64)
    nd = dgrad(x).shape[0]
    X = np.zeros((nd,) + x.shape)
    f0, g0, G0, d, D = Inf, None, None, None, None
    x_start, X_start = None, None
    alpha, evals, evals_tan, iters, status = 0.0, 0, 0, 0, 0
    best = None
    lbfgs = LBFGSDiff(mem)
    t0 = optm.elapsed_time(0.0)
    stage = 0
    while True:
        if stage == 0:
            act = x > lower
            x = np.maximum(x, lower)
        else:
            z = x_start + alpha * d
            act = z > lower
            x = np.maximum(z, lower)
            X = act * (X_start + alpha * D)
        f, g = fg(x)
        g = np.asarray(g, dtype=np.float64)
        evals += 1
        if best is None or f < best[0]:
            best = (f, g, x, X)
        if stage != 0:
            lnsrch.iterate(f)
            if lnsrch.converged():
                iters += 1
                stage = 2
            else:
                alpha = lnsrch.step()
                stage = 1
        if stage != 1:
            freevars = optm.unblocked_variables(x, lower, None, g)
            pg = freevars * g
            pgnorm = optm.norm2(pg)
            if evals == 1:
                gtest = max(gatol, grtol * pgnorm)
            if pgnorm <= gtest:
                status = optm.GTEST_SATISFIED
                break
            if stage == 2:
                if f <= fatol or abs(f - f0) <= frtol * max(abs(f), abs(f0)):
                    status = optm.FTEST_SATISFIED
                    break
                s = x - x_start
                snorm = optm.norm2(s)
                if snorm <= xatol or (xrtol > 0 and snorm <= xrtol * optm.norm2(x)):
                    status = optm.XTEST_SATISFIED
                    break
            if iters >= maxiter:
                status = optm.TOO_MANY_ITERATIONS
                break
        if evals >= maxeval:
            status = optm.TOO_MANY_EVALUATIONS
            break
        if stage != 1:
            # tangent of the gradient at the new iterate: G_j = H(x) X_j + d g / d mu_j
            G = np.array(dgrad(x), dtype=np.float64)
            for j in range(nd):
                xn = np.linalg.norm(X[j])
                if xn == 0:
                    continue
                h = hvp_step * max(np.linalg.norm(x), 1e-30) / xn
                if central:
                    gp = np.asarray(fg(x + h * X[j])[1], dtype=np.float64)
                    gm = np.asarray(fg(x - h * X[j])[1], dtype=np.float64)
                    G[j] += (gp - gm) / (2 * h)
                    evals_tan += 2
                else:
                    G[j] += (np.asarray(fg(x + h * X[j])[1], dtype=np.float64) - g) / h
                    evals_tan += 1
            if verb > 0 and iters % verb == 0:
                print(f"{iters:7d} {1e3 * optm.elapsed_time(t0):11.3f} {evals:7d} {evals_tan:7d} {f:23.15e} "
                      f"{pgnorm:11.3e} {alpha:11.3e}  |X| {' '.join(f'{np.linalg.norm(X[j]):.3e}' for j in range(nd))}",
                      file=output, flush=True)
            if stage != 0:
                if freeze_metric:
                    lbfgs.update(x - x_start, g - g0, np.zeros_like(X), np.zeros_like(X))
                else:
                    lbfgs.update(x - x_start, g - g0, X - X_start, G - G0)
            d, D, scaled = lbfgs.apply(-g, -G, freevars)
            flg = 2
            dg = _dot(d, g)
            if not scaled:
                flg = 1
            elif dg >= 0:
                d, D = -pg, -(freevars * G)
                dg = -pgnorm ** 2
                flg = 1
            if flg == 2:
                alpha = 1.0
            else:
                alpha = optm.steepest_descent_step(x, pgnorm, f, f2nd=None, fmin=None, dxrel=None, dxabs=1.0)
            amax = optm.line_search_step_max(x, lower, None, 1, d)
            alpha = min(alpha, amax)
            lnsrch.start(f, dg, alpha)
            stage = 1
            f0, x_start, X_start, g0, G0 = f, x, X, g, G
    if best[0] < f:
        f, g, x, X = best
    if verb > 0:
        print(f"# Termination: {optm.reason(status)} ({evals} evaluations, {evals_tan} for the tangents)", file=output,
              flush=True)
    return x, f, g, status, X, dict(evals=evals, evals_tan=evals_tan, iters=iters)


def sugar_iterative(gD, g_delta, X, X_eps):
    """d MC-SURE / d mu_j = grad D(x_hat)^T X_j + g_delta^T (X_eps_j - X_j), and its two terms."""
    nd = X.shape[0]
    dD = np.array([_dot(gD, X[j]) for j in range(nd)])
    d2div = np.array([_dot(g_delta, X_eps[j] - X[j]) for j in range(nd)])
    return dD + d2div, dict(dD=dD, d2div=d2div)
