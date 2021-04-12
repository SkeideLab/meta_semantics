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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook #03: Comparison to Adult Meta-Analysis

# %%
from os import makedirs
from shutil import copy

from IPython.display import display
from nilearn import image, plotting, reporting
from nimare import io

from nb01_ale import run_ale
from nb02_subtraction import run_subtraction

# %%
# Copy Sleuth text files to the results folder
makedirs("../results/adults", exist_ok=True)
copy("../results/ale/all.txt", "../results/adults/children.txt")
copy("../data/adults/adults.txt", "../results/adults/adults.txt")

# Read Sleuth text files for children and adults
dset1 = io.convert_sleuth_to_dataset("../results/adults/children.txt")
dset2 = io.convert_sleuth_to_dataset("../results/adults/adults.txt")

# %%
# Perform the ALE analysis for adults
run_ale(
    text_file="../results/adults/adults.txt",
    voxel_thresh=0.001,
    cluster_thresh=0.01,
    random_seed=1234,
    n_iters=1000,
    output_dir="../results/adults",
)

# %%
# Perform subtraction analysis for children vs. adults
run_subtraction(
    text_file1="../results/adults/children.txt",
    text_file2="../results/adults/adults.txt",
    voxel_thresh=0.001,
    cluster_size=200,
    random_seed=1234,
    n_iters=10000,
    output_dir="../results/adults",
)

# %%
# Glass brain for adults only
img = image.load_img("../results/adults/adults_z_thresh.nii.gz")
p = plotting.plot_glass_brain(img, display_mode="lyrz", colorbar=True)

# Table for adults only
t = reporting.get_clusters_table(img, stat_threshold=0, min_distance=1000)
display(t)

# %%
# Glass brain for children vs. adults
img_sub = image.load_img("../results/adults/children_minus_adults_z_thresh.nii.gz")
p_sub = plotting.plot_glass_brain(
    img_sub,
    display_mode="lyrz",
    colorbar=True,
    vmax=4,
    plot_abs=False,
    symmetric_cbar=True,
)

# Table brain for children vs. adults
t_sub = reporting.get_clusters_table(
    img_sub, stat_threshold=0, min_distance=1000, two_sided=True
)
display(t_sub)
