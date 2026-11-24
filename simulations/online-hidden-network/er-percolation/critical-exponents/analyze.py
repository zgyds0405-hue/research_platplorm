"""Estimate order-parameter exponents with paired independent-graph bootstrap."""
import csv

import numpy as np
from scipy.optimize import brentq
from scipy.stats import t
import xlsxwriter

from run_critical import ROOT, probabilities

BOOTSTRAPS = 3000
BETA_WINDOWS = ((.08, .20), (.06, .16), (.10, .24))
GROUPS = ("Static", "Markov")


def theory(epsilon):
    """Return k*S_infinity for ER site percolation at k*p = 1 + epsilon."""
    if epsilon <= 0:
        return 0.0
    return brentq(lambda u: -(1 + epsilon) * np.expm1(-u) - u,
                  1e-9, 1 + epsilon, xtol=1e-14)


def slope_weights(x, correction=None):
    columns = [np.ones(len(x)), x]
    if correction is not None:
        columns.append(correction)
    return np.linalg.pinv(np.column_stack(columns))[1]


def interval(values):
    return np.quantile(values, [.025, .975])


def load():
    dataset = np.load(ROOT / "paired_runs.npz")
    sizes, degrees = dataset["sizes"], dataset["degrees"]
    reps = int(dataset["replicates"])
    snapshots = int(dataset["snapshots"])
    values = {(int(k), int(n)): dataset[f"k{k}_n{n}"][..., [0, 2]] / (n * snapshots)
              for k in degrees for n in sizes}
    return sizes, degrees, reps, values


def compute():
    sizes, degrees, reps, values = load()
    generator = np.random.default_rng(927155)
    margin = t.ppf(.975, reps - 1) / np.sqrt(reps)
    scan_rows, beta_rows, size_rows, differences = [], [], [], []
    for k in degrees:
        critical_means, critical_boot = [], []
        for n in sizes:
            samples = values[k, n]
            probs = probabilities(n, k)
            epsilon = k * probs - 1
            scaled = epsilon * np.cbrt(n / k)
            mean = samples.mean(axis=0)
            half = samples.std(axis=0, ddof=1) * margin
            delta = samples[:, :, 1] - samples[:, :, 0]
            for j, p in enumerate(probs):
                scan_rows.append((k,n,p,epsilon[j],scaled[j],mean[j,0],half[j,0],mean[j,1],half[j,1],
                                  delta[:,j].mean(),delta[:,j].std(ddof=1)*margin,theory(epsilon[j])/k))
            weights = generator.multinomial(reps, np.ones(reps) / reps, size=BOOTSTRAPS) / reps
            center, = np.flatnonzero(epsilon == 0)
            critical_means.append(mean[center])
            critical_boot.append(weights @ samples[:, center])
            if n not in sizes[-2:]:
                continue
            for lo, hi in BETA_WINDOWS:
                select = (epsilon >= lo) & (epsilon <= hi)
                dx = epsilon[select]
                measured = mean[select] * k
                boot_measured = np.einsum("br,rpa->bpa", weights, samples[:, select]) * k
                for label, correction in (("Log-log", None), ("First-order correction", dx)):
                    coef = slope_weights(np.log(dx), correction)
                    estimates = coef @ np.log(measured)
                    boot = np.einsum("p,bpa->ba", coef, np.log(boot_measured))
                    theoretical = coef @ np.log([theory(v) for v in dx])
                    for i, group in enumerate(GROUPS):
                        beta_rows.append((k,n,lo,hi,dx.size,label,group,estimates[i],
                                          *interval(boot[:,i]),theoretical))
                    differences.append(("Beta",k,n,lo,hi,label,estimates[1]-estimates[0],
                                        *interval(boot[:,1]-boot[:,0])))
        for start in (0, 1):
            coef = slope_weights(np.log(sizes[start:]))
            exponent = coef @ np.log(critical_means[start:])
            boot = np.einsum("n,nba->ba", coef, np.log(critical_boot[start:]))
            for i, group in enumerate(GROUPS):
                size_rows.append((k,sizes[start],sizes[-1],group,exponent[i],*interval(boot[:,i]),-1/3))
            differences.append(("Size",k,sizes[-1],sizes[start],sizes[-1],"Log-log",
                                exponent[1]-exponent[0],*interval(boot[:,1]-boot[:,0])))
    return [
        ("Critical scan", "near_critical",
         "k N p epsilon scaled_x static_S static_95_half dynamic_S dynamic_95_half dynamic_minus_static paired_95_half theory_S",scan_rows),
        ("Beta fits", "beta_fits",
         "k N epsilon_min epsilon_max points fit group beta ci95_low ci95_high exact_curve_same_fit",beta_rows),
        ("At criticality", "size_fits",
         "k min_N max_N group exponent ci95_low ci95_high theory",size_rows),
        ("Paired differences", "paired_fits",
         "observable k max_N window_min window_max fit dynamic_minus_static ci95_low ci95_high",differences),
    ]


def main():
    with xlsxwriter.Workbook(ROOT / "critical_exponents.xlsx") as book:
        header = book.add_format({"bold":True, "bg_color":"#19435F", "font_color":"white"})
        fmt = book.add_format({"num_format":"0.000000"})
        for title, stem, names, rows in compute():
            cols = names.split()
            with (ROOT / f"{stem}.csv").open("w", newline="") as out:
                writer = csv.writer(out, lineterminator="\n")
                writer.writerow(cols)
                writer.writerows(rows)
            sheet = book.add_worksheet(title)
            sheet.write_row(0,0,cols,header)
            sheet.freeze_panes(1,2)
            for j, name in enumerate(cols):
                integers = name in {"k","N","points","min_N","max_N"}
                sheet.set_column(j,j,max(13,len(name)+4),None if integers else fmt)
            for i,row in enumerate(rows,1):
                sheet.write_row(i,0,row)
            sheet.autofilter(0,0,len(rows),len(cols)-1)


if __name__ == "__main__":
    main()
