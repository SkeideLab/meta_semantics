# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ![SkeideLab and MPI CBS logos](../misc/header_logos.png)
#
# # Notebook #08: Output Figures
#
# *Created April 2021 by Alexander Enge* ([enge@cbs.mpg.de](mailto:enge@cbs.mpg.de))
#
# This notebooks contains the code that we've used to create publication-ready figures from the statistical maps that we've created in the previous notebooks. We did not add any explanatory text because no substantial work is happening here and the solutions we've used are very idiosyncratic to the present meta-analysis. Also, most of what is happening should hopefully become clear from the code itself and the comments. For most figures, we rely heavily on Nilearn's `plot_glass_brain()` function and combine multiple of these glass brains using matplotlib's `gridspec` syntax. Note that there is no code for Figure 1 (showing the literature search and selection process in the form of a [PRISMA flowchart](https://doi.org/10.1136/bmj.n71)). This is because this figure was created manually using [LibreOffice Impress](https://www.libreoffice.org).

# %%
from os import makedirs

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

# Convert to categories and sort
tasks = ["Knowledge", "Relatedness", "Objects"]
exps["task_type"] = exps["task_type"].astype("category")
exps["task_type"].cat.reorder_categories(tasks, inplace=True)

# Get numeric codes for task types
tasks_int = exps["task_type"].cat.codes

# Create a custom color palette for the task types
alpha = 0.8
vmax_viridis = 2 / 0.75
norm = mpl.colors.Normalize(vmin=0, vmax=vmax_viridis, clip=True)
mapper = mpl.cm.ScalarMappable(norm=norm, cmap="viridis")
rgbas = [mapper.to_rgba(color) for color in range(3)]
palette = dict(zip(tasks, rgbas))

# Create a dummy plot so we can create a legend from it later on
p = sns.scatterplot(
    data=exps,
    x="n",
    y="n",
    hue="task_type",
    palette=palette,
    hue_order=tasks,
)
handles, labels = plt.gca().get_legend_handles_labels()
_ = [lh.set_linewidth(0) for lh in handles]
_ = [lh.set_alpha(alpha) for lh in handles]

# %%
# Create Figure 3 (study-level descriptive variables)
figsize = (90 * scaling, 90 * scaling)
fig3, axs = plt.subplots(nrows=3, ncols=3, figsize=figsize)

# Specify custom margins
_ = fig3.subplots_adjust(
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
        alpha=alpha,
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
    tasks_colors = [palette[tasks[x]] for x in tasks_int]
    plt.axis(xmin=xbins.min(), xmax=xbins.max(), ymin=ybins.min(), ymax=ybins.max())
    p = sns.regplot(
        data=exps,
        x=x,
        y=y,
        color="gray",
        scatter_kws=dict({"linewidths": 0, "color": tasks_colors, "alpha": alpha}),
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
bins_peaks = np.arange(0, 51, step=5)
bins_age = np.arange(4, 14, step=1)

# Create the histograms
histplot_custom(x="n", bins=bins_n, ax=axs[0][0])
histplot_custom(x="peaks_n", bins=bins_peaks, ax=axs[1][1])
histplot_custom(x="age_mean", bins=bins_age, ax=axs[2][2])

# Create the regression plots
regplot_custom(x="n", y="peaks_n", ax=axs[1][0], xbins=bins_n, ybins=bins_peaks)
regplot_custom(x="n", y="age_mean", ax=axs[2][0], xbins=bins_n, ybins=bins_age)
regplot_custom(
    x="peaks_n", y="age_mean", ax=axs[2][1], xbins=bins_peaks, ybins=bins_age
)

# Add the regression coefficients
plot_reg_coef(x="peaks_n", y="n", ax=axs[0][1])
plot_reg_coef(x="age_mean", y="n", ax=axs[0][2])
plot_reg_coef(x="age_mean", y="peaks_n", ax=axs[1][2])

# Add the legend
bbox = (0.9, 0.2, 0, 0)
axs[0][2].legend(handles, labels, bbox_to_anchor=bbox, frameon=False, handlelength=0.3)

# Set axis labels to the outer plots
axs[2][0].set_xlabel("Sample size")
axs[2][1].set_xlabel("No. of peaks")
axs[2][2].set_xlabel("Mean age")
axs[0][0].set_ylabel("Sample size")
axs[1][0].set_ylabel("No. of peaks")
axs[2][0].set_ylabel("Mean age")

# Save as PDF
fig3.savefig("../results/figures/fig3.pdf")

# %%
# Extract all individual peaks and their z score
peaks_coords = np.array(exps["peaks_mni"].explode().tolist())
peaks_zstat = [
    stats.norm.ppf(stats.t.cdf(tstat, df=n - 1)) if tstat else np.nan
    for tstat, n in zip(exps.explode("tstats")["tstats"], exps.explode("tstats")["n"])
]

# Get indices of peaks without an effect size
idxs_p = np.where(np.isnan(peaks_zstat))[0]

# Create Figure 4 (main ALE and SDM analyses)
figsize = (90 * scaling, 145 * scaling)
fig4 = plt.figure(figsize=figsize)
gs = fig4.add_gridspec(145, 90)
ax1 = fig4.add_subplot(gs[0:35, :])
ax2 = fig4.add_subplot(gs[35:70, :])
ax3 = fig4.add_subplot(gs[70:95, :])
ax4 = fig4.add_subplot(gs[95:120, :])
ax5 = fig4.add_subplot(gs[120:145, :])

# Specify smaller margins
margins = {
    "left": 0,
    "bottom": 0,
    "right": 1 - 0.1 * scaling / figsize[0],
    "top": 1 - 0.1 * scaling / figsize[1],
}
_ = fig4.subplots_adjust(**margins)

# Plot individual peaks (without effect sizes)
p1_1 = plotting.plot_markers(  # left and right
    node_values=[0.5] * len(idxs_p),
    node_coords=np.take(peaks_coords, idxs_p, axis=0),
    node_size=10,
    node_cmap="binary",
    node_vmin=0,
    node_vmax=1,
    alpha=alpha,
    display_mode="lr",
    axes=ax1,
    node_kwargs=dict({"linewidths": 0}),
    colorbar=False,
)
p2_1 = plotting.plot_markers(  # coronal and horizontal
    node_values=[0.5] * len(idxs_p),
    node_coords=np.take(peaks_coords, idxs_p, axis=0),
    node_size=10,
    node_cmap="binary",
    node_vmin=0,
    node_vmax=1,
    alpha=alpha,
    display_mode="yz",
    axes=ax2,
    node_kwargs=dict({"linewidths": 0}),
    colorbar=False,
)

# Plot individual peaks (with effect sizes)
vmin, vmax = 0, 8
p1_2 = plotting.plot_markers(  # left and right
    node_values=np.delete(peaks_zstat, idxs_p).astype("float"),
    node_coords=np.delete(peaks_coords, idxs_p, axis=0),
    node_size=10,
    node_cmap="YlOrRd",
    node_vmin=vmin,
    node_vmax=vmax,
    alpha=alpha,
    display_mode="lr",
    axes=ax1,
    node_kwargs=dict({"linewidths": 0}),
    colorbar=False,
)
p2_2 = plotting.plot_markers(  # coronal and horizontal
    node_values=np.delete(peaks_zstat, idxs_p).astype("float"),
    node_coords=np.delete(peaks_coords, idxs_p, axis=0),
    node_size=10,
    node_cmap="YlOrRd",
    node_vmin=vmin,
    node_vmax=vmax,
    alpha=alpha,
    display_mode="yz",
    axes=ax2,
    node_kwargs=dict({"linewidths": 0}),
    colorbar=False,
)

# Plot z-maps from ALE
img_all = image.load_img("../results/ale/all_z_thresh.nii.gz")
p3 = plotting.plot_glass_brain(None, display_mode="lyrz", axes=ax3)
p3.add_overlay(img_all, cmap="YlOrRd", vmin=vmin, vmax=vmax)

# Plot thresholded Z-map from SDM analysis without covariates
img_sdm = image.load_img(
    "../results/sdm/analysis_mod1/mod1_z_voxelCorrected_p_0.00100_50.nii.gz"
)
p4 = plotting.plot_glass_brain(None, display_mode="lyrz", axes=ax4)
p4.add_overlay(img_sdm, cmap="YlOrRd", vmin=vmin, vmax=vmax)

# Plot thresholded Z-map from SDM analysis with covariates
img_sdm = image.load_img(
    "../results/sdm/analysis_mod2/mod2_z_voxelCorrected_p_0.00100_50.nii.gz"
)
p5 = plotting.plot_glass_brain(None, display_mode="lyrz", axes=ax5)
p5.add_overlay(img_sdm, cmap="YlOrRd", vmin=vmin, vmax=vmax)

# Add a joint colorbar
ax_cbar = fig4.add_subplot(gs[42:64, 43:46])
cmap = plt.get_cmap("YlOrRd")
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap, norm=norm, label="$\it{z}$ score")
plt.axhline(y=3.1, color="black", linewidth=1)

# Add subplot labels
_ = ax1.annotate("A", xy=(0, 0.95), xycoords="axes fraction", weight="bold")
_ = ax3.annotate("B", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax4.annotate("C", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax5.annotate("D", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax1.annotate("Peaks", xy=(0.035, 0.95), xycoords="axes fraction")
_ = ax3.annotate("ALE", xy=(0.035, 0.96), xycoords="axes fraction")
_ = ax4.annotate("SDM", xy=(0.035, 0.96), xycoords="axes fraction")
_ = ax5.annotate("SDM + covariates", xy=(0.035, 0.96), xycoords="axes fraction")

# Save to PDF
fig4.savefig("../results/figures/fig4.pdf")

# %%
# Get task types of individual peaks
peaks_tasks = exps.explode("tstats")["task_type"].cat.codes

# Create Figure 5 (task category-specific ALEs)
figsize = (90 * scaling, 145 * scaling)
fig5 = plt.figure(figsize=figsize)
gs = fig5.add_gridspec(145, 90)
ax1 = fig5.add_subplot(gs[0:35, :])
ax2 = fig5.add_subplot(gs[35:70, :])
ax3 = fig5.add_subplot(gs[70:95, :])
ax4 = fig5.add_subplot(gs[95:120, :])
ax5 = fig5.add_subplot(gs[120:145, :])

# Specify smaller margins
_ = fig5.subplots_adjust(**margins)

# Plot individual peaks
p1 = plotting.plot_markers(  # left and right
    node_values=peaks_tasks,
    node_coords=peaks_coords,
    node_size=10,
    node_cmap="viridis",
    node_vmin=0,
    node_vmax=vmax_viridis,
    alpha=alpha,
    display_mode="lr",
    axes=ax1,
    node_kwargs=dict({"linewidths": 0}),
    colorbar=False,
)
p2 = plotting.plot_markers(  # coronal and horizontal
    node_values=peaks_tasks,
    node_coords=peaks_coords,
    node_size=10,
    node_cmap="viridis",
    node_vmin=0,
    node_vmax=vmax_viridis,
    alpha=alpha,
    display_mode="yz",
    axes=ax2,
    node_kwargs=dict({"linewidths": 0}),
    colorbar=False,
)

# Plot z maps for the task-specific ALEs
for task, ax_task in zip(["knowledge", "relatedness", "objects"], [ax3, ax4, ax5]):
    img_task = image.load_img("../results/ale/" + task + "_z_thresh.nii.gz")
    p_task = plotting.plot_glass_brain(None, display_mode="lyrz", axes=ax_task)
    p_task.add_overlay(img_task, cmap="YlOrRd", vmin=vmin, vmax=vmax)

# Add a legend for the task types
ax_leg = fig5.add_subplot(gs[33, 57])
ax_leg.legend(handles, labels, frameon=False, handlelength=0.5)
ax_leg.set_axis_off()

# Add a joint colorbar for the ALE maps
ax_cbar = fig5.add_subplot(gs[48:70, 43:46])
mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap, norm=norm, label="$\it{z}$ score")
plt.axhline(y=3.1, color="black", linewidth=1)

# Add subplot labels
_ = ax1.annotate("A", xy=(0, 0.95), xycoords="axes fraction", weight="bold")
_ = ax3.annotate("B", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax4.annotate("C", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax5.annotate("D", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax1.annotate("Peaks", xy=(0.035, 0.95), xycoords="axes fraction")
_ = ax3.annotate("Knowledge", xy=(0.035, 0.96), xycoords="axes fraction")
_ = ax4.annotate("Relatedness", xy=(0.035, 0.96), xycoords="axes fraction")
_ = ax5.annotate("Objects", xy=(0.035, 0.96), xycoords="axes fraction")

# Save to PDF
fig5.savefig("../results/figures/fig5.pdf")

# %%
# Create Figure 6 (differences between task categories)
figsize = (90 * scaling, 87 * scaling)
fig6 = plt.figure(figsize=figsize)
gs = fig6.add_gridspec(85, 90)
ax1 = fig6.add_subplot(gs[1:26, :])
ax2 = fig6.add_subplot(gs[26:51, :])
ax3 = fig6.add_subplot(gs[51:76, :])

# Specify smaller margins
_ = fig6.subplots_adjust(**margins)

# Plot z maps for the subtraction analyses
vmin, vmid, vmax = -4, 0, 4
for task, ax_sub in zip(["knowledge", "relatedness", "objects"], [ax1, ax2, ax3]):
    img_sub = image.load_img(
        "../results/subtraction/" + task + "_minus_n" + task + "_z_thresh.nii.gz"
    )
    _ = plotting.plot_glass_brain(
        img_sub,
        display_mode="lyrz",
        axes=ax_sub,
        cmap="RdYlBu_r",
        vmin=vmid,
        vmax=vmax,
        plot_abs=False,
        symmetric_cbar=True,
    )

# Add colorbar
ax_cbar = fig6.add_subplot(gs[75:78, 36:54])
cmap = plt.get_cmap("RdYlBu_r")
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
mpl.colorbar.ColorbarBase(
    ax_cbar,
    cmap=cmap,
    norm=norm,
    orientation="horizontal",
    ticks=np.arange(vmin, vmax + 1, 2),
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
fig6.savefig("../results/figures/fig6.pdf")

# %%
# Create Figure 7 (age-related changes)
figsize = (90 * scaling, 62 * scaling)
fig7 = plt.figure(figsize=figsize)
gs = fig7.add_gridspec(60, 90)
ax1 = fig7.add_subplot(gs[1:26, :])
ax2 = fig7.add_subplot(gs[26:51, :])

# Specify smaller margins
_ = fig7.subplots_adjust(**margins)

# Plot z map for ALE median split
p1 = plotting.plot_glass_brain(
    "../results/subtraction/older_minus_younger_z_thresh.nii.gz",
    display_mode="lyrz",
    axes=ax1,
    cmap="RdYlBu_r",
    vmin=vmid,
    vmax=vmax,
    plot_abs=False,
    symmetric_cbar=True,
)

# Plot z map for SDM meta-regression
p2 = plotting.plot_glass_brain(
    "../results/sdm/analysis_mod3/mod3_z.nii.gz",
    display_mode="lyrz",
    axes=ax2,
    cmap="RdYlBu_r",
    vmin=vmid,
    vmax=vmax,
    plot_abs=False,
    symmetric_cbar=True,
)

# Add colorbar
ax_cbar = fig7.add_subplot(gs[50:53, 36:54])
cmap = plt.get_cmap("RdYlBu_r")
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
mpl.colorbar.ColorbarBase(
    ax_cbar,
    cmap=cmap,
    norm=norm,
    orientation="horizontal",
    ticks=np.arange(vmin, vmax + 1, 2),
    label="$\it{z}$ score",
)
plt.axvline(x=-3.1, color="black", linewidth=1)
plt.axvline(x=3.1, color="black", linewidth=1)

# Add subplot labels
_ = ax1.annotate("A", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax2.annotate("B", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax1.annotate(
    "ALE median split (older > younger)", xy=(0.035, 0.96), xycoords="axes fraction"
)
_ = ax2.annotate(
    "SDM meta-regression (uncorrected)", xy=(0.035, 0.96), xycoords="axes fraction"
)

# Save to PDF
fig7.savefig("../results/figures/fig7.pdf")

# %%
# Create Figure 8 (comparison with adults)
figsize = (90 * scaling, 95 * scaling)
fig8 = plt.figure(figsize=figsize)
gs = fig8.add_gridspec(96, 90)
ax1 = fig8.add_subplot(gs[1:26, :])
ax2 = fig8.add_subplot(gs[31:56, :])
ax3 = fig8.add_subplot(gs[61:86, :])

# Specify smaller margins
_ = fig8.subplots_adjust(**margins)

# Plot ALE map for adults
vmin_1, vmax_1 = 0, 12
img_adults = image.load_img("../results/adults/adults_z_thresh.nii.gz")
p1 = plotting.plot_glass_brain(None, display_mode="lyrz", axes=ax1)
p1.add_overlay(img_adults, cmap="YlOrRd", vmin=vmin_1, vmax=vmax_1)

# Plot children > adults & adults > children
vmin_2, vmid_2, vmax_2 = -4, 0, 4
img_sub = image.load_img("../results/adults/children_minus_adults_z_thresh.nii.gz")
p2 = plotting.plot_glass_brain(
    img_sub,
    display_mode="lyrz",
    axes=ax2,
    cmap="RdYlBu_r",
    vmin=vmid_2,
    vmax=vmax_2,
    plot_abs=False,
    symmetric_cbar=True,
)

# Plot conjunction
vmin_3, vmax_3 = 0, 8
img_conj = image.load_img("../results/adults/children_conj_adults_z.nii.gz")
p3 = plotting.plot_glass_brain(None, display_mode="lyrz", axes=ax3)
p3.add_overlay(img_conj, cmap="YlOrRd", vmin=vmin_3, vmax=vmax_3)

# Add colorbar for adults
ax1_cbar = fig8.add_subplot(gs[25:28, 36:54])
cmap1 = plt.get_cmap("YlOrRd")
norm = mpl.colors.Normalize(vmin=vmin_1, vmax=vmax_1)
mpl.colorbar.ColorbarBase(
    ax1_cbar,
    cmap=cmap1,
    norm=norm,
    orientation="horizontal",
    ticks=np.arange(vmin_1, vmax_1 + 1, 4),
)
plt.axvline(x=3.1, color="black", linewidth=1)

# Add colorbar for children > adults & adults > children
ax2_cbar = fig8.add_subplot(gs[55:58, 36:54])
cmap2 = plt.get_cmap("RdYlBu_r")
norm = mpl.colors.Normalize(vmin=vmin_2, vmax=vmax_2)
mpl.colorbar.ColorbarBase(
    ax2_cbar,
    cmap=cmap2,
    norm=norm,
    orientation="horizontal",
    ticks=np.arange(vmin_2, vmax_2 + 1, 2),
)
plt.axvline(x=-3.1, color="black", linewidth=1)
plt.axvline(x=3.1, color="black", linewidth=1)

# Add colorbar for conjunction
ax3_cbar = fig8.add_subplot(gs[85:88, 36:54])
norm = mpl.colors.Normalize(vmin=vmin_3, vmax=vmax_3)
mpl.colorbar.ColorbarBase(
    ax3_cbar,
    cmap=cmap1,
    norm=norm,
    orientation="horizontal",
    ticks=np.arange(vmin_3, vmax_3 + 1, 2),
    label="$\it{z}$ score",
)
plt.axvline(x=3.1, color="black", linewidth=1)

# Add subplot labels
_ = ax1.annotate("A", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax2.annotate("B", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax3.annotate("C", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax1.annotate("Adults", xy=(0.035, 0.96), xycoords="axes fraction")
_ = ax2.annotate("Children > adults", xy=(0.035, 0.96), xycoords="axes fraction")
_ = ax3.annotate("Conjunction", xy=(0.035, 0.96), xycoords="axes fraction")

# Save to PDF
fig8.savefig("../results/figures/fig8.pdf")

# %%
# Create Figure 9 (leave-one-out analysis)
figsize = (90 * scaling, 110 * scaling)
fig9 = plt.figure(figsize=figsize)
gs = fig9.add_gridspec(110, 90)
ax1 = fig9.add_subplot(gs[1:26, :])
ax2 = fig9.add_subplot(gs[26:51, :])
ax3 = fig9.add_subplot(gs[51:76, :])
ax4 = fig9.add_subplot(gs[76:101, :])

# Specify smaller margins
_ = fig9.subplots_adjust(**margins)

# Plot mean jackknife reliability maps
vmin, vmax = 0, 100
for task, ax_jk in zip(
    ["all", "knowledge", "relatedness", "objects"], [ax1, ax2, ax3, ax4]
):
    img_task = image.load_img("../results/ale/" + task + "_z_thresh.nii.gz")
    img_mask = image.math_img("np.where(img > 0, 100, 0)", img=img_task)
    img_jk = image.load_img("../results/jackknife/" + task + "/mean_jk.nii.gz")
    img_jk = image.math_img("img1 * img2", img1=img_jk, img2=img_mask)
    p = plotting.plot_glass_brain(None, display_mode="lyrz", axes=ax_jk)
    p.add_overlay(img_jk, cmap="RdYlGn", vmin=vmin, vmax=vmax)

# Add colorbar
ax_cbar = fig9.add_subplot(gs[100:103, 36:54])
cmap = plt.get_cmap("RdYlGn")
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
mpl.colorbar.ColorbarBase(
    ax_cbar,
    cmap=cmap,
    norm=norm,
    orientation="horizontal",
    ticks=[vmin, vmax, (vmin + vmax) / 2],
    label="Leave-one-out robustness (%)",
)

# Add subplot labels
_ = ax1.annotate("A", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax2.annotate("B", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax3.annotate("C", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax4.annotate("D", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax1.annotate("All semantic cognition", xy=(0.035, 0.96), xycoords="axes fraction")
_ = ax2.annotate("Knowledge", xy=(0.035, 0.96), xycoords="axes fraction")
_ = ax3.annotate("Relatedness", xy=(0.035, 0.96), xycoords="axes fraction")
_ = ax4.annotate("Objects", xy=(0.035, 0.96), xycoords="axes fraction")

# Save to PDF
fig9.savefig("../results/figures/fig9.pdf")

# %%
# Create Figure 10 (fail-safe N analaysis)
figsize = (90 * scaling, 110 * scaling)
fig10 = plt.figure(figsize=figsize)
gs = fig10.add_gridspec(110, 90)
ax1 = fig10.add_subplot(gs[1:26, :])
ax2 = fig10.add_subplot(gs[26:51, :])
ax3 = fig10.add_subplot(gs[51:76, :])
ax4 = fig10.add_subplot(gs[76:101, :])

# Specify smaller margins
_ = fig10.subplots_adjust(**margins)

# Plot mean FSN maps (scaled by the number of original experiments)
vmin, vmax = 0, 60
n_studies_per_task = [len(exps)] + exps["task_type"].value_counts().to_list()
for task, ax_fsn, n_studies in zip(
    ["all", "knowledge", "relatedness", "objects"],
    [ax1, ax2, ax3, ax4],
    n_studies_per_task,
):
    img_task = image.load_img("../results/ale/" + task + "_z_thresh.nii.gz")
    img_mask = image.math_img("np.where(img > 0, 100, 0)", img=img_task)
    img_fsn = image.load_img("../results/fsn/" + task + "/" + task + "_mean_fsn.nii.gz")
    formula = "img1 * img2 / " + str(n_studies)
    img_perc = image.math_img(formula=formula, img1=img_fsn, img2=img_mask)
    p = plotting.plot_glass_brain(None, display_mode="lyrz", axes=ax_fsn)
    p.add_overlay(img_perc, cmap="RdYlGn", vmin=vmin, vmax=vmax)

# Add colorbar
ax_cbar = fig10.add_subplot(gs[100:103, 36:54])
cmap = plt.get_cmap("RdYlGn")
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
mpl.colorbar.ColorbarBase(
    ax_cbar,
    cmap=cmap,
    norm=norm,
    orientation="horizontal",
    ticks=[vmin, vmax, (vmin + vmax) / 2],
    label="Fail-safe N (%)",
)

# Add subplot labels
_ = ax1.annotate("A", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax2.annotate("B", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax3.annotate("C", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax4.annotate("D", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax1.annotate("All semantic cognition", xy=(0.035, 0.96), xycoords="axes fraction")
_ = ax2.annotate("Knowledge", xy=(0.035, 0.96), xycoords="axes fraction")
_ = ax3.annotate("Relatedness", xy=(0.035, 0.96), xycoords="axes fraction")
_ = ax4.annotate("Objects", xy=(0.035, 0.96), xycoords="axes fraction")

# Save to PDF
fig10.savefig("../results/figures/fig10.pdf")
