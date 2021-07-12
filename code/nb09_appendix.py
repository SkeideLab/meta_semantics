# -*- coding: utf-8 -*-
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
# # Notebook #09: Appendix
#
# *Created July 2021 by Alexander Enge* ([enge@cbs.mpg.de](mailto:enge@cbs.mpg.de))
#
# Here we perform additional ALE analyses (see Notebook #01) and subtraction analyses (see Notebook #02) to explore the effect of four different experiment-level covariates, namely (a) the language used in the original experiment (alphabetic vs. logograpic), (b) the modality of stimulus presentation (visual vs. auditory/audiovisual), (c) the modality of children's response (manual vs. overt/covert/none), and (d) the toolbox used for fMRI analysis in the original article (SPM vs. other).

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn import plotting
from scipy.stats import norm

from nb01_ale import run_ale, write_peaks_to_sleuth
from nb02_subtraction import run_subtraction

# %%
# Read table of experiments from ALE analysis
exps = pd.read_json("../results/exps.json")

# %%
# Define ALE analyses for four different covariates
basedir = "../results/appendix"
alphabetic_languages = ["english", "dutch", "german", "french"]
ales = dict(
    {
        f"{basedir}/alphabetic.txt": f"language == {alphabetic_languages}",
        f"{basedir}/nalphabetic.txt": f"language != {alphabetic_languages}",
        f"{basedir}/visual.txt": 'modality_pres == "visual"',
        f"{basedir}/nvisual.txt": 'modality_pres != "visual"',
        f"{basedir}/manual.txt": 'modality_resp == "manual"',
        f"{basedir}/nmanual.txt": 'modality_resp != "manual"',
        f"{basedir}/spm.txt": 'software == "SPM"',
        f"{basedir}/nspm.txt": 'software != "SPM"',
    }
)

# Count number of experiments per ALE
ns = [len(exps.query(value)) for value in ales.values()]

# Write the relevant experiments to Sleuth text files
for key, value in zip(ales.keys(), ales.values()):
    write_peaks_to_sleuth(text_file=key, df=exps, query=value)

# %%
# Compute the ALE for each Sleuth file
for key in ales.keys():
    run_ale(
        text_file=key,
        voxel_thresh=0.001,
        cluster_thresh=0.01,
        random_seed=1234,
        n_iters=1000,
        output_dir="../results/appendix/",
    )

# %%
# Define subraction analyses
subtrs = dict(
    {
        f"{basedir}/alphabetic.txt": f"{basedir}/nalphabetic.txt",
        f"{basedir}/visual.txt": f"{basedir}/nvisual.txt",
        f"{basedir}/manual.txt": f"{basedir}/nmanual.txt",
        f"{basedir}/spm.txt": f"{basedir}/nspm.txt",
    }
)

# Run subtraction analyses
for key, value in zip(subtrs.keys(), subtrs.values()):
    run_subtraction(
        text_file1=key,
        text_file2=value,
        voxel_thresh=0.001,
        cluster_size_mm3=200,
        random_seed=1234,
        n_iters=20000,
        output_dir="../results/appendix",
    )

# %%
# Create empty figure with subplots
scaling = 2 / 30
figsize = (90 * scaling, 110 * scaling)
fig = plt.figure(figsize=figsize)
gs = fig.add_gridspec(110, 90)
ax1 = fig.add_subplot(gs[1:26, :])
ax2 = fig.add_subplot(gs[26:51, :])
ax3 = fig.add_subplot(gs[51:76, :])
ax4 = fig.add_subplot(gs[76:101, :])

# Smaller margins
margins = {
    "left": 0,
    "bottom": 0,
    "right": 1 - 0.1 * scaling / figsize[0],
    "top": 1 - 0.1 * scaling / figsize[1],
}
_ = fig.subplots_adjust(**margins)

# Plot subtraction maps
covariates = ["alphabetic", "visual", "manual", "spm"]
basedir = "../results/appendix"
vmin = 0
vmax = 5
for covariate, ax in zip(covariates, [ax1, ax2, ax3, ax4]):
    fname_unthresh = f"{basedir}/{covariate}_minus_n{covariate}_z.nii.gz"
    p = plotting.plot_glass_brain(
        fname_unthresh,
        display_mode="lyrz",
        axes=ax,
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax,
        plot_abs=False,
        symmetric_cbar=True,
    )

# Add subplot labels
_ = ax1.annotate("A", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax2.annotate("B", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax3.annotate("C", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax4.annotate("D", xy=(0, 0.96), xycoords="axes fraction", weight="bold")
_ = ax1.annotate(
    "Alphabetic > logographic language",
    xy=(0.035, 0.96),
    xycoords="axes fraction",
)
_ = ax2.annotate(
    "Visual > auditory/audiovisual stimuli",
    xy=(0.035, 0.96),
    xycoords="axes fraction",
)
_ = ax3.annotate(
    "Manual > varbal/no response",
    xy=(0.035, 0.96),
    xycoords="axes fraction",
)
_ = ax4.annotate(
    "SPM > other analysis software",
    xy=(0.035, 0.96),
    xycoords="axes fraction",
)

# Add colorbar
ax_cbar = fig.add_subplot(gs[100:103, 36:54])
cmap = plt.get_cmap("RdYlBu_r")
norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
mpl.colorbar.ColorbarBase(
    ax_cbar,
    cmap=cmap,
    norm=norm,
    orientation="horizontal",
    ticks=np.arange(-vmax, vmax + 1, 2),
    label="$\it{z}$ score",
)

# Save to PDF
fig.savefig("../results/figures/fig11.pdf")
