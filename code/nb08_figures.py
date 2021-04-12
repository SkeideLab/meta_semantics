# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: 'Python 3.6.13 64-bit (''nimare'': conda)'
#     metadata:
#       interpreter:
#         hash: 4514b7299076417b8e187233bf4c34d62c86c82f88944b861e2dca408b3b9212
#     name: python3
# ---

# %%
from os import makedirs

import font_source_sans_pro
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn import image, plotting
from scipy import stats
from scipy.stats import pearsonr

# %%
# Specify default font
mpl.rcParams.update({"font.family": ["Liberation Sans"], "font.size": 12})

# Create output directory
makedirs("../results/figures", exist_ok=True)

# Define scaling factor so we can set all the figure sizes in scaling
scaling = 2 / 30

# %%
# Read table of experiments from ALE analysis
exps = pd.read_json("../results/exps.json")

# Capitalize the task type labels
exps["task_type"] = exps["task_type"].str.capitalize()

# Get numeric indices for task types
tasks, tasks_int = np.unique(exps["task_type"], return_inverse=True)

# Create a custom color palette for the task types
vmax_viridis = 2 / 0.75
norm = mpl.colors.Normalize(vmin=0, vmax=vmax_viridis, clip=True)
mapper = mpl.cm.ScalarMappable(norm=norm, cmap="viridis")
rgbas = [mapper.to_rgba(color) for color in range(3)]
palette = dict(zip(tasks, rgbas))

# Create a dummy plot so we can create a legend from it later on
p = sns.scatterplot(
    data=exps, x="n", y="n", hue="task_type", palette=palette, hue_order=tasks
)
handles, labels = plt.gca().get_legend_handles_labels()

# %%
# Create a new figure for the distributions of sample size, no. of foci, and mean age
figsize = (90 * scaling, 90 * scaling)
fig2, axs = plt.subplots(nrows=3, ncols=3, figsize=figsize)

# Specify custom margins
_ = fig2.subplots_adjust(
    left=8 * scaling / figsize[0],
    bottom=8 * scaling / figsize[1],
    right=1 - 1 * scaling / figsize[0],
    top=1 - 1 * scaling / figsize[1],
)

# Create a custom function for the histograms
def histplot_custom(x, bins, ax):
    plt.sca(ax)
    plt.axis(xmin=bins.min(), xmax=bins.max())
    p = sns.histplot(
        data=exps,
        x=x,
        bins=bins,
        hue="task_type",
        hue_order=tasks,
        palette=palette,
        multiple="stack",
        legend=False,
    )
    p.set_xticks(bins[0::2])
    p.set_ylabel("")
    p.set_xlabel("")
    p.tick_params(axis="y", colors="white")


# Create a custom function for the regression plots
def regplot_custom(x, y, ax, xbins, ybins):
    plt.sca(ax)
    taks_colors = [palette[tasks[x]] for x in tasks_int]
    plt.axis(xmin=xbins.min(), xmax=xbins.max(), ymin=ybins.min(), ymax=ybins.max())
    p = sns.regplot(
        data=exps,
        x=x,
        y=y,
        color="gray",
        scatter_kws=dict({"linewidths": 0, "color": taks_colors}),
    )
    p.set_xticks(xbins[0::2])
    p.set_yticks(ybins[0::2])
    p.set_ylabel("")
    p.set_xlabel("")


# Create a custom function to show the regression coefficient
def plot_reg_coef(x, y, ax):
    plt.sca(ax)
    r, p = pearsonr(exps[x], exps[y])
    label = "r = {:.2f}".format(r).replace("0.", ".").replace("r", "$\it{r}$")
    plt.gca().annotate(label, xy=(0.5, 0.5), xycoords="axes fraction", ha="center")
    plt.gca().set_axis_off()


# Specify separate bin widths for each variable
bins_n = np.arange(0, 80, step=10)
bins_foci = np.arange(0, 51, step=5)
bins_age = np.arange(4, 14, step=1)

# Create the histograms
histplot_custom(x="n", bins=bins_n, ax=axs[0][0])
histplot_custom(x="foci_n", bins=bins_foci, ax=axs[1][1])
histplot_custom(x="age_mean", bins=bins_age, ax=axs[2][2])

# Create the regression plots
regplot_custom(x="n", y="foci_n", ax=axs[1][0], xbins=bins_n, ybins=bins_foci)
regplot_custom(x="n", y="age_mean", ax=axs[2][0], xbins=bins_n, ybins=bins_age)
regplot_custom(x="foci_n", y="age_mean", ax=axs[2][1], xbins=bins_foci, ybins=bins_age)

# Add the regression coefficients
plot_reg_coef(x="foci_n", y="n", ax=axs[0][1])
plot_reg_coef(x="age_mean", y="n", ax=axs[0][2])
plot_reg_coef(x="age_mean", y="foci_n", ax=axs[1][2])

# Add the legend
bbox = (0.9, 0.2, 0, 0)
axs[0][2].legend(handles, labels, bbox_to_anchor=bbox, frameon=False, handlelength=0.5)

# Set axis labels to the outer plots
axs[2][0].set_xlabel("Sample size")
axs[2][1].set_xlabel("No. of foci")
axs[2][2].set_xlabel("Mean age")
axs[0][0].set_ylabel("Sample size")
axs[1][0].set_ylabel("No. of foci")
axs[2][0].set_ylabel("Mean age")

# Save as PDF
fig2.savefig("../results/figures/fig2.pdf")

# %%
# Extract all individual foci and their z-value
foci_coords = np.array(exps["foci_mni"].explode().tolist())
foci_zstat = [
    stats.norm.ppf(stats.t.cdf(tstat, df=n - 1)) if tstat else np.nan
    for tstat, n in zip(exps.explode("tstats")["tstats"], exps.explode("tstats")["n"])
]

# Get indices of foci without an effect size
idxs_p = np.where(np.isnan(foci_zstat))[0]

# Create a new figure with four subplots
figsize = (90 * scaling, 145 * scaling)
fig3 = plt.figure(figsize=figsize)
gs = fig3.add_gridspec(145, 90)
ax1 = fig3.add_subplot(gs[0:35, :])
ax2 = fig3.add_subplot(gs[35:70, :])
ax3 = fig3.add_subplot(gs[70:95, :])
ax4 = fig3.add_subplot(gs[95:120, :])
ax5 = fig3.add_subplot(gs[120:145, :])

# Specify smaller margins
margins = {
    "left": 0,
    "bottom": 0,
    "right": 1 - 0.1 * scaling / figsize[0],
    "top": 1 - 0.1 * scaling / figsize[1],
}
_ = fig3.subplots_adjust(**margins)

# Plot individual foci (without effect sizes)
p1_1 = plotting.plot_markers(  # left and right
    node_values=[0.5] * len(idxs_p),
    node_coords=np.take(foci_coords, idxs_p, axis=0),
    node_size=10,
    node_cmap="binary",
    node_vmin=0,
    node_vmax=1,
    display_mode="lr",
    axes=ax1,
    node_kwargs=dict({"linewidths": 0}),
    colorbar=False,
)
p2_1 = plotting.plot_markers(  # coronal and horizontal
    node_values=[0.5] * len(idxs_p),
    node_coords=np.take(foci_coords, idxs_p, axis=0),
    node_size=10,
    node_cmap="binary",
    node_vmin=0,
    node_vmax=1,
    display_mode="yz",
    axes=ax2,
    node_kwargs=dict({"linewidths": 0}),
    colorbar=False,
)

# Plot individual foci (with effect sizes)
p1_2 = plotting.plot_markers(  # left and right
    node_values=np.delete(foci_zstat, idxs_p).astype("float"),
    node_coords=np.delete(foci_coords, idxs_p, axis=0),
    node_size=10,
    node_cmap="YlOrRd",
    node_vmin=0,
    node_vmax=8,
    display_mode="lr",
    axes=ax1,
    node_kwargs=dict({"linewidths": 0}),
    colorbar=False,
)
p2_2 = plotting.plot_markers(  # coronal and horizontal
    node_values=np.delete(foci_zstat, idxs_p).astype("float"),
    node_coords=np.delete(foci_coords, idxs_p, axis=0),
    node_size=10,
    node_cmap="YlOrRd",
    node_vmin=0,
    node_vmax=8,
    display_mode="yz",
    axes=ax2,
    node_kwargs=dict({"linewidths": 0}),
    colorbar=False,
)

# Plot z-maps from ALE
img_all = image.load_img("../results/ale/all_z_thresh.nii.gz")
p3 = plotting.plot_glass_brain(None, display_mode="lyrz", axes=ax3)
p3.add_overlay(img_all, cmap="YlOrRd", vmin=0, vmax=8)

# # Plot non-cluster thresholded map below
# img = image.load_img("../results/ale/all_z.nii.gz")
# thresh = image.load_img("../results/ale/all_z_thresh.nii.gz")
# mask = image.math_img("np.where(img > 0, 1, 0)", img=thresh)
# p3 = plotting.plot_glass_brain(None, display_mode="lyrz", axes=ax3)
# p3.add_overlay(img, cmap="YlOrRd", vmin=0, vmax=8)
# p3.add_contours(mask, colors="black", linewidths=0.1, vmin=1, vmax=1)

# Plot thresholded Z-map from SDM analysis without covariates
img_sdm = image.load_img(
    "../results/sdm/analysis_mod1/mod1_z_voxelCorrected_p_0.00100_50.nii.gz"
)
p4 = plotting.plot_glass_brain(None, display_mode="lyrz", axes=ax4)
p4.add_overlay(img_sdm, cmap="YlOrRd", vmin=0, vmax=8)

# Plot thresholded Z-map from SDM analysis with covariates
img_sdm = image.load_img(
    "../results/sdm/analysis_mod2/mod2_z_voxelCorrected_p_0.00100_50.nii.gz"
)
p5 = plotting.plot_glass_brain(None, display_mode="lyrz", axes=ax5)
p5.add_overlay(img_sdm, cmap="YlOrRd", vmin=0, vmax=8)

# Add a joint colorbar
ax_cbar = fig3.add_subplot(gs[42:64, 43:46])
cmap = plt.get_cmap("YlOrRd")
norm = mpl.colors.Normalize(vmin=0, vmax=8)
mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap, norm=norm, label="$\it{z}$ score")
plt.axhline(y=3.1, color="black", linewidth=1)

# Add subplot labels
_ = ax1.annotate("A", xy=(0, 0.95), xycoords="axes fraction", weight="bold")
_ = ax3.annotate("B", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax4.annotate("C", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax5.annotate("D", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax1.annotate("Foci", xy=(0.035, 0.95), xycoords="axes fraction")
_ = ax3.annotate("ALE", xy=(0.035, 0.96), xycoords="axes fraction")
_ = ax4.annotate("SDM", xy=(0.035, 0.96), xycoords="axes fraction")
_ = ax5.annotate("SDM + covariates", xy=(0.035, 0.96), xycoords="axes fraction")

# Save to PDF
fig3.savefig("../results/figures/fig3.pdf")

# %%
# Get task types of individual foci
foci_tasks = exps.explode("tstats")["task_type"]
_, foci_tasks = np.unique(foci_tasks, return_inverse=True)

# Create a new figure with four subplots
figsize = (90 * scaling, 145 * scaling)
fig4 = plt.figure(figsize=figsize)
gs = fig4.add_gridspec(145, 90)
ax1 = fig4.add_subplot(gs[0:35, :])
ax2 = fig4.add_subplot(gs[35:70, :])
ax3 = fig4.add_subplot(gs[70:95, :])
ax4 = fig4.add_subplot(gs[95:120, :])
ax5 = fig4.add_subplot(gs[120:145, :])

# Specify smaller margins
_ = fig4.subplots_adjust(**margins)

# Plot individual foci
p1 = plotting.plot_markers(  # left and right
    node_values=foci_tasks,
    node_coords=foci_coords,
    node_size=10,
    node_cmap="viridis",
    node_vmin=0,
    node_vmax=vmax_viridis,
    display_mode="lr",
    axes=ax1,
    node_kwargs=dict({"linewidths": 0}),
    colorbar=False,
)
p2 = plotting.plot_markers(  # coronal and horizontal
    node_values=foci_tasks,
    node_coords=foci_coords,
    node_size=10,
    node_cmap="viridis",
    node_vmin=0,
    node_vmax=vmax_viridis,
    display_mode="yz",
    axes=ax2,
    node_kwargs=dict({"linewidths": 0}),
    colorbar=False,
)

# Plot z maps for the task-specific ALEs
for task, ax_task in zip(["knowledge", "relatedness", "objects"], [ax3, ax4, ax5]):
    img_task = image.load_img("../results/ale/" + task + "_z_thresh.nii.gz")
    p_task = plotting.plot_glass_brain(None, display_mode="lyrz", axes=ax_task)
    p_task.add_overlay(img_task, cmap="YlOrRd", vmin=0, vmax=8)

# Add a legend for the task types
ax_leg = fig4.add_subplot(gs[33, 56])
ax_leg.legend(handles, labels, frameon=False, handlelength=0.5)
ax_leg.set_axis_off()

# Add a joint colorbar for the ALE maps
ax_cbar = fig4.add_subplot(gs[48:70, 43:46])
mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap, norm=norm, label="$\it{z}$ score")
plt.axhline(y=3.1, color="black", linewidth=1)

# Add subplot labels
_ = ax1.annotate("A", xy=(0, 0.95), xycoords="axes fraction", weight="bold")
_ = ax3.annotate("B", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax4.annotate("C", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax5.annotate("D", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax1.annotate("Foci", xy=(0.035, 0.95), xycoords="axes fraction")
_ = ax3.annotate("Knowledge", xy=(0.035, 0.96), xycoords="axes fraction")
_ = ax4.annotate("Relatedness", xy=(0.035, 0.96), xycoords="axes fraction")
_ = ax5.annotate("Objects", xy=(0.035, 0.96), xycoords="axes fraction")

# Save to PDF
fig4.savefig("../results/figures/fig4.pdf")

# %%
# Create a new figure with four subplots
figsize = (90 * scaling, 87 * scaling)
fig5 = plt.figure(figsize=figsize)
gs = fig5.add_gridspec(85, 90)
ax1 = fig5.add_subplot(gs[1:26, :])
ax2 = fig5.add_subplot(gs[26:51, :])
ax3 = fig5.add_subplot(gs[51:76, :])

# Specify smaller margins
_ = fig5.subplots_adjust(**margins)

# Plot z maps for the subtraction analyses
for task, ax_sub in zip(["knowledge", "relatedness", "objects"], [ax1, ax2, ax3]):
    img_sub = image.load_img(
        "../results/subtraction/" + task + "_minus_n" + task + "_z_thresh.nii.gz"
    )
    _ = plotting.plot_glass_brain(
        img_sub,
        display_mode="lyrz",
        axes=ax_sub,
        cmap="RdYlBu_r",
        vmin=0,
        vmax=4,
        plot_abs=False,
        symmetric_cbar=True,
    )

# Add colorbar
ax_cbar = fig5.add_subplot(gs[75:78, 36:54])
cmap = plt.get_cmap("RdYlBu_r")
norm = mpl.colors.Normalize(vmin=-4, vmax=4)
mpl.colorbar.ColorbarBase(
    ax_cbar,
    cmap=cmap,
    norm=norm,
    orientation="horizontal",
    ticks=np.arange(-4, 5, 2),
    label="$\it{z}$ score",
)
plt.axvline(x=-3.1, color="black", linewidth=1)
plt.axvline(x=3.1, color="black", linewidth=1)

# Add subplot labels
_ = ax1.annotate("A", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax2.annotate("B", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax3.annotate("C", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax1.annotate("Knowledge > other", xy=(0.035, 0.96), xycoords="axes fraction")
_ = ax2.annotate("Relatedness > other", xy=(0.035, 0.96), xycoords="axes fraction")
_ = ax3.annotate("Objects > other", xy=(0.035, 0.96), xycoords="axes fraction")

# Save to PDF
fig5.savefig("../results/figures/fig5.pdf")

# %%
# Create a new figure with three subplots
figsize = (90 * scaling, 95 * scaling)
fig6 = plt.figure(figsize=figsize)
gs = fig6.add_gridspec(95, 90)
ax1 = fig6.add_subplot(gs[0:25, :])
ax2 = fig6.add_subplot(gs[30:55, :])
ax3 = fig6.add_subplot(gs[60:85, :])

# Specify smaller margins
_ = fig6.subplots_adjust(**margins)

# Plot ALE map for adults
img_adults = image.load_img("../results/adults/adults_z_thresh.nii.gz")
p1 = plotting.plot_glass_brain(None, display_mode="lyrz", axes=ax1)
p1.add_overlay(img_adults, cmap="YlOrRd", vmin=0, vmax=12)

# Plot children > adults & adults > children
img_sub = image.load_img("../results/adults/children_minus_adults_z_thresh.nii.gz")
p2 = plotting.plot_glass_brain(
    img_sub,
    display_mode="lyrz",
    axes=ax2,
    cmap="RdYlBu_r",
    vmin=0,
    vmax=4,
    plot_abs=False,
    symmetric_cbar=True,
)

# Plot conjunction
img_conj = image.load_img("../results/adults/children_conj_adults_z.nii.gz")
p3 = plotting.plot_glass_brain(None, display_mode="lyrz", axes=ax3)
p3.add_overlay(img_conj, cmap="YlGn", vmin=0, vmax=8)

# Add colorbar for adults
ax1_cbar = fig6.add_subplot(gs[24:27, 36:54])
cmap1 = plt.get_cmap("YlOrRd")
norm = mpl.colors.Normalize(vmin=0, vmax=12)
mpl.colorbar.ColorbarBase(
    ax1_cbar, cmap=cmap1, norm=norm, orientation="horizontal", ticks=np.arange(0, 13, 4)
)
plt.axvline(x=3.1, color="black", linewidth=1)

# Add colorbar for children > adults & adults > children
ax2_cbar = fig6.add_subplot(gs[54:57, 36:54])
cmap2 = plt.get_cmap("RdYlBu_r")
norm = mpl.colors.Normalize(vmin=-4, vmax=4)
mpl.colorbar.ColorbarBase(
    ax2_cbar, cmap=cmap2, norm=norm, orientation="horizontal", ticks=np.arange(-4, 5, 2)
)
plt.axvline(x=-3.1, color="black", linewidth=1)
plt.axvline(x=3.1, color="black", linewidth=1)

# Add colorbar for conjunction
ax3_cbar = fig6.add_subplot(gs[84:87, 36:54])
cmap3 = plt.get_cmap("YlGn")
norm = mpl.colors.Normalize(vmin=0, vmax=8)
mpl.colorbar.ColorbarBase(
    ax3_cbar,
    cmap=cmap3,
    norm=norm,
    orientation="horizontal",
    ticks=np.arange(0, 9, 2),
    label="$\it{z}$ score",
)
plt.axvline(x=3.1, color="black", linewidth=1)

# Add colorbar labels
for row, col, label in zip(
    [26, 56, 56, 86],
    [31, 31, 73, 31],
    ["Adults", "Adults > children", "Children > adults", "Conjunction"],
):
    ax1_cbar_label = fig6.add_subplot(gs[row : row + 2, col + 2])
    ax1_cbar_label.text(0, 0, label, horizontalalignment="right")
    ax1_cbar_label.set_axis_off()

# Add subplot labels
_ = ax1.annotate("A", xy=(0, 0.9), xycoords="axes fraction", weight="bold")
_ = ax2.annotate("B", xy=(0, 0.9), xycoords="axes fraction", weight="bold")
_ = ax3.annotate("C", xy=(0, 0.9), xycoords="axes fraction", weight="bold")

# Save to PDF
fig6.savefig("../results/figures/fig6.pdf")
