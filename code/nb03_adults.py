# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] pycharm={"name": "#%% md\n"}
# # Notebook #03: Comparison to Adult Meta-Analysis

# %%
# Import modules
from os import makedirs
from shutil import copy
from nimare import io
from nb01_ale import run_ale
from nb02_subtraction import run_subtraction
from nilearn import image, plotting, reporting

# %%

# Copy Sleuth text files to the results folder
makedirs('../results/adults', exist_ok=True)
copy('../results/ale/all.txt', '../results/adults/children.txt')
copy('../data/adults/adults.txt', '../results/adults/adults.txt')

# Read Sleuth text files for children and adults
dset1 = io.convert_sleuth_to_dataset('../results/adults/children.txt')
dset2 = io.convert_sleuth_to_dataset('../results/adults/adults.txt')

# %%
# Perform the ALE analysis for adults
run_ale(text_file='../results/adults/adults.txt', voxel_thresh=0.001, cluster_thresh=0.01,
        n_iters=1000, output_dir='../results/adults')

# %%
# Perform subtraction analysis for children vs. adults
run_subtraction(text_file1='../results/adults/children.txt',
                text_file2='../results/adults/adults.txt',
                voxel_thresh=0.01, cluster_size=200, n_iters=10000,
                output_dir='../results/adults')

# %%

# Output for adults only
img = image.load_img('../results/adults/adults_z_thresholded.nii.gz')
p = plotting.plot_glass_brain(img, display_mode='lyrz', colorbar=True)
t = reporting.get_clusters_table(img, stat_threshold=0, min_distance=1000)
t.style.format({'X': '{:.0f}', 'Y': '{:.0f}', 'Z': '{:.0f}', 'Peak Stat': '{:.2f}'}).hide_index()

# %%

# Output for children vs. adults
img = image.load_img('../results/adults/children_minus_adults_z_tresholded.nii.gz')
p = plotting.plot_glass_brain(img, display_mode='lyrz', colorbar=True, symmetric_cbar=True)
t = reporting.get_clusters_table(img, stat_threshold=0, min_distance=1000)
t.style.format({'X': '{:.0f}', 'Y': '{:.0f}', 'Z': '{:.0f}', 'Peak Stat': '{:.2f}'}).hide_index()
