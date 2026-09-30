import pathlib
import numpy as np

# Summary of the convergence tests of run_results/convergence_test*.py: for each tolerance (ftol:xtol), objective f,
# number of evaluations, time, and distance of x to the reconstruction obtained with the strictest tolerance
# (reference = smallest xtol among ftol = 0); PSNR for the synthetic configurations.
ROOT = pathlib.Path(__file__).resolve().parents[2]

for folder in sorted((ROOT / "results" / "convergence").glob("*")):
    runs = [dict(np.load(f, allow_pickle=True)) for f in folder.glob("ftol*_xtol*.npz")]
    if not runs:
        continue
    runs.sort(key=lambda r: (-float(r["ftol"]), -float(r["xtol"])))   # from the loosest to the strictest
    ref = min((r for r in runs if float(r["ftol"]) == 0), key=lambda r: float(r["xtol"]), default=runs[-1])
    xr, fr = ref["x"], float(ref["fx"])
    print(f"\n=== {folder.name}  (reference: ftol={float(ref['ftol']):g}, xtol={float(ref['xtol']):g})")
    print(f"  {'ftol':>6s} {'xtol':>6s} {'f - f_ref':>12s} {'evals':>7s} {'time [h]':>8s} {'status':>6s} "
          f"{'||x-x_ref||/||x_ref||':>22s} {'max x':>9s}" + ("  PSNR whole / support" if "psnr_whole" in ref else ""))
    for r in runs:
        rel = np.linalg.norm(r["x"] - xr) / np.linalg.norm(xr)
        line = (f"  {float(r['ftol']):6g} {float(r['xtol']):6g} {float(r['fx']) - fr:12.4e} {int(r['n_eval']):7d} "
                f"{float(r['time']) / 3600:8.2f} {str(r['status']):>6s} {rel:22.4f} {r['x'].max():9.2e}")
        if "psnr_whole" in r:
            line += f"  {float(r['psnr_whole']):6.3f} / {float(r['psnr_supp']):6.3f}"
        print(line)
