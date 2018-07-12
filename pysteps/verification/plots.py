
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import numpy as np

def plot_reldiag(reldiag, ax=None):
    if ax is None:
        ax = plt.gca()
    
    f = 1.0 * reldiag["Y_sum"] / reldiag["num_idx"]
    r = 1.0 * reldiag["X_sum"] / reldiag["num_idx"]
    
    mask = np.logical_and(np.isfinite(r), np.isfinite(f))
    
    ax.plot(r[mask], f[mask], "kD-")
    ax.plot([0, 1], [0, 1], "k--")
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    ax.grid(True, ls=':')
    
    ax.set_xlabel("Forecast probability")
    ax.set_ylabel("Observed relative frequency")
    
    # 
    iax = inset_axes(ax, width="35%", height="20%", loc=4, borderpad=3.5)
    bw = reldiag["bin_edges"][2] - reldiag["bin_edges"][1]
    iax.bar(reldiag["bin_edges"][:-1], reldiag["sample_size"], width=bw, 
            align="edge", color="gray", edgecolor="black")
    iax.set_yscale("log", basey=10)
    iax.set_xticks(reldiag["bin_edges"])
    iax.set_xticklabels(["%.1f" % max(v, 1e-6) for v in reldiag["bin_edges"]])
    yt_min = int(max(np.floor(np.log10(min(reldiag["sample_size"][:-1]))), 1))
    yt_max = int(np.ceil(np.log10(max(reldiag["sample_size"][:-1]))))
    t = [pow(10.0, k) for k in range(yt_min, yt_max)]
    
    iax.set_yticks([int(t_) for t_ in t])
    iax.set_xlim(0.0, 1.0)
    iax.set_ylim(t[0], 5*t[-1])
    iax.set_ylabel("log10(samples)")
    iax.yaxis.tick_right()
    iax.yaxis.set_label_position("right")
    iax.tick_params(axis="both", which="major", labelsize=6)
