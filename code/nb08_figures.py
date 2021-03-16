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
import pandas as pd
import numpy as np
from nilearn import image, plotting
import matplotlib as mpl
from scipy import stats
from os import makedirs

# %%
# Set fonts for matplotlib
mpl.rcParams.update({"font.family": ["FreeSans"], "font.size": 15})

# Create output directory
makedirs("../results/figures", exist_ok=True)

# We want to be able to specify the figure size in mm
mm = 0.1 / 2.54

# %%
# Read table of experiments from ALE analysis
exps = pd.read_json("../results/exps.json")

# Extract all individual foci and their z-value
foci_coords = np.array(exps["foci_mni"].explode().tolist())
foci_zstat = [
    stats.norm.ppf(stats.t.cdf(tstat, df=n - 1)) if tstat else np.nan
    for tstat, n in zip(exps.explode("tstats")["tstats"], exps.explode("tstats")["n"])
]

# Get indices of foci without an effect size
idxs_p = np.where(np.isnan(foci_zstat))[0]

# Create a new figure with four subplots
figsize = (190 * mm, 265 * mm)
fig2 = mpl.pyplot.figure(figsize=figsize)
gs = fig2.add_gridspec(255, 180)
ax1 = fig2.add_subplot(gs[0:60, :])
ax2 = fig2.add_subplot(gs[60:120, :])
ax3 = fig2.add_subplot(gs[120:165, :])
ax4 = fig2.add_subplot(gs[165:210, :])
ax5 = fig2.add_subplot(gs[210:255, :])

# Specify smaller margins
margin = 5 * mm
margins = {
    "left": margin / figsize[0],
    "bottom": margin / figsize[1],
    "right": 1 - margin / figsize[0],
    "top": 1 - margin / figsize[1],
}
_ = fig2.subplots_adjust(**margins)

# Add subplot labels
_ = ax1.annotate("A", xy=(0, 0.9), xycoords="axes fraction", weight="bold")
_ = ax3.annotate("B", xy=(0, 0.9), xycoords="axes fraction", weight="bold")
_ = ax4.annotate("C", xy=(0, 0.9), xycoords="axes fraction", weight="bold")
_ = ax5.annotate("D", xy=(0, 0.9), xycoords="axes fraction", weight="bold")

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
ax_cbar = fig2.add_subplot(gs[65:110, 86:92])
cmap = mpl.pyplot.get_cmap("YlOrRd")
norm = mpl.colors.Normalize(vmin=0, vmax=8)
mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap, norm=norm, label="$\it{z}$ score")
mpl.pyplot.axhline(y=3.1, color="black", linewidth=1)

# Save to PDF
fig2.savefig("../results/figures/fig2.pdf")

# %%
# Get task types of individual foci
foci_tasks = exps.explode("tstats")["task_type"]
foci_tasks_lookup, foci_types_int = np.unique(foci_tasks, return_inverse=True)

# Create a new figure with four subplots
figsize = (190 * mm, 265 * mm)
fig3 = mpl.pyplot.figure(figsize=figsize)
gs = fig3.add_gridspec(255, 180)
ax1 = fig3.add_subplot(gs[0:60, :])
ax2 = fig3.add_subplot(gs[60:120, :])
ax3 = fig3.add_subplot(gs[120:165, :])
ax4 = fig3.add_subplot(gs[165:210, :])
ax5 = fig3.add_subplot(gs[210:255, :])

# Specify smaller margins
_ = fig3.subplots_adjust(**margins)

# Add subplot labels
_ = ax1.annotate("A", xy=(0, 0.9), xycoords="axes fraction", weight="bold")
_ = ax3.annotate("B", xy=(0, 0.9), xycoords="axes fraction", weight="bold")
_ = ax4.annotate("C", xy=(0, 0.9), xycoords="axes fraction", weight="bold")
_ = ax5.annotate("D", xy=(0, 0.9), xycoords="axes fraction", weight="bold")

# Plot individual foci
p1 = plotting.plot_markers(  # left and right
    node_values=foci_types_int,
    node_coords=foci_coords,
    node_size=10,
    node_cmap="viridis",
    node_vmin=0,
    node_vmax=2 / 0.85,
    display_mode="lr",
    axes=ax1,
    node_kwargs=dict({"linewidths": 0}),
    colorbar=False,
)
p2 = plotting.plot_markers(  # coronal and horizontal
    node_values=foci_types_int,
    node_coords=foci_coords,
    node_size=10,
    node_cmap="viridis",
    node_vmin=0,
    node_vmax=2 / 0.85,
    display_mode="yz",
    axes=ax2,
    node_kwargs=dict({"linewidths": 0}),
    colorbar=False,
)

# Plot z maps for the task-specific ALEs
for task, ax_task in zip(["knowledge", "lexical", "objects"], [ax3, ax4, ax5]):
    img_task = image.load_img("../results/ale/" + task + "_z_thresh.nii.gz")
    p_task = plotting.plot_glass_brain(None, display_mode="lyrz", axes=ax_task)
    p_task.add_overlay(img_task, cmap="YlOrRd", vmin=0, vmax=8)

# Add a legend for the task types
legend_elements = [
    mpl.lines.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Knowledge",
        markerfacecolor="#440154FF",
        markersize=8,
    ),
    mpl.lines.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Lexical",
        markerfacecolor="#277E8EFF",
        markersize=8,
    ),
    mpl.lines.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Objects",
        markerfacecolor="#9AD93CFF",
        markersize=8,
    ),
]
ax_leg = fig3.add_subplot(gs[65:100, 94])
ax_leg.legend(handles=legend_elements, loc="center", frameon=False, handlelength=0)
ax_leg.set_axis_off()

# Add a joint colorbar for the ALE maps
ax_cbar = fig3.add_subplot(gs[65:110, 165:171])
cmap = mpl.pyplot.get_cmap("YlOrRd")
norm = mpl.colors.Normalize(vmin=0, vmax=8)
mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap, norm=norm, label="$\it{z}$ score")
mpl.pyplot.axhline(y=3.1, color="black", linewidth=1)

# Save to PDF
fig3.savefig("../results/figures/fig3.pdf")

# %%
# Create a new figure with three subplots
figsize = (190 * mm, 145 * mm)
fig4 = mpl.pyplot.figure(figsize=figsize)
gs = fig4.add_gridspec(135, 180)
ax1 = fig4.add_subplot(gs[0:45, :])
ax2 = fig4.add_subplot(gs[45:90, :])
ax3 = fig4.add_subplot(gs[90:135, :])

# Specify smaller margins
_ = fig4.subplots_adjust(**margins)

# Add subplot labels
_ = ax1.annotate("A", xy=(0, 0.9), xycoords="axes fraction", weight="bold")
_ = ax2.annotate("B", xy=(0, 0.9), xycoords="axes fraction", weight="bold")
_ = ax3.annotate("C", xy=(0, 0.9), xycoords="axes fraction", weight="bold")

# Plot ALE map for adults
img_adults = image.load_img("../results/adults/adults_z_thresh.nii.gz")
p1 = plotting.plot_glass_brain(None, display_mode="lyrz", axes=ax1)
p1.add_overlay(img_adults, cmap="YlOrRd", vmin=0, vmax=12)

# Plot children > adults & adults > children
img_diff = image.load_img("../results/adults/children_minus_adults_z_thresh.nii.gz")
p2 = plotting.plot_glass_brain(
    img_diff,
    display_mode="lyrz",
    axes=ax2,
    cmap="RdYlBu_r",
    vmin=0,
    vmax=4,
    plot_abs=False,
    symmetric_cbar=True,
)

# Plot conjunction
formula = "np.where(img1 * img2 > 0, np.minimum(img1, img2), 0)"
img_conj = image.math_img(formula, img1=img_all, img2=img_adults)
p3 = plotting.plot_glass_brain(None, display_mode="lyrz", axes=ax3)
p3.add_overlay(img_conj, cmap="YlGn", vmin=0, vmax=8)

# # Add colorbars
# ax1_cbar = fig4.add_subplot(gs[19:21, 15:17])
# cmap = plotting.cm._cmap_d['black_red_r']
# norm = mpl.colors.Normalize(vmin=0, vmax=12)
# mpl.colorbar.ColorbarBase(ax1_cbar, cmap=cmap, norm=norm, label='$\it{Z}$-score', orientation='horizontal')
# mpl.pyplot.axvline(x=3.1, color='black', linestyle=":", linewidth=1)

# ax2_cbar = fig4.add_subplot(gs[46:48, 14:18])
# cmap = plotting.cm._cmap_d['cold_white_hot']
# norm = mpl.colors.Normalize(vmin=-5, vmax=5)
# mpl.colorbar.ColorbarBase(ax2_cbar, cmap=cmap, norm=norm, label='Children > adults | Adults > children', orientation='horizontal')
# mpl.pyplot.axvline(x=-3.1, color='black', linestyle=":", linewidth=1)
# mpl.pyplot.axvline(x=3.1, color='black', linestyle=":", linewidth=1)

# ax3_cbar = fig4.add_subplot(gs[73:75, 15:17])
# cmap = plotting.cm._cmap_d['black_green_r']
# norm = mpl.colors.Normalize(vmin=0, vmax=0.06)
# mpl.colorbar.ColorbarBase(ax3_cbar, cmap=cmap, norm=norm, label='ALE value', orientation='horizontal')

# Save to PDF
fig4.savefig("../results/figures/fig4.pdf")
