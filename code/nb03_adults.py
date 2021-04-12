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
from nibabel import save
from nilearn import image, plotting, reporting
from nimare import io

from nb01_ale import run_ale
from nb02_subtraction import run_subtraction

# %%
# Copy Sleuth text files to the results folder
output_dir = "../results/adults/"
_ = makedirs(output_dir, exist_ok=True)
_ = copy("../results/ale/all.txt", output_dir + "children.txt")
_ = copy("../data/adults/adults.txt", output_dir + "adults.txt")

# %%
# Perform the ALE analysis for adults
_ = run_ale(
    text_file=output_dir + "adults.txt",
    voxel_thresh=0.001,
    cluster_thresh=0.01,
    random_seed=1234,
    n_iters=1000,
    output_dir=output_dir,
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
    output_dir=output_dir,
)

# %%
# Compute seperate difference maps for children > adults and adults > children
img_sub = image.load_img(output_dir + "children_minus_adults_z_thresh.nii.gz")
img_children_gt_adults = image.math_img("np.where(img > 0, img, 0)", img = img_sub)
img_adults_gt_children = image.math_img("np.where(img < 0, img * -1, 0)", img = img_sub)
_ = save(img_children_gt_adults, output_dir + "children_greater_adults_z_thresh.nii.gz")
_ = save(img_adults_gt_children, output_dir + "adults_greater_children_z_thresh.nii.gz")

# %%
# Compute conjunction z map (= minimum voxel-wise z score across both groups)
formula = "np.where(img1 * img2 > 0, np.minimum(img1, img2), 0)"
img_adults_z = image.load_img(output_dir + "adults_z_thresh.nii.gz")
img_children_z = image.load_img("../results/ale/all_z_thresh.nii.gz")
img_conj_z = image.math_img(formula, img1=img_adults_z, img2=img_children_z)
_ = save(img_conj_z, output_dir + "children_conj_adults_z.nii.gz")

# Compute conjunction ALE map (= minimum voxel-wise ALE value across both groups)
img_adults_ale = image.load_img(output_dir + "adults_stat_thresh.nii.gz")
img_children_ale = image.load_img("../results/ale/all_stat_thresh.nii.gz")
img_conj_ale = image.math_img(formula, img1=img_adults_ale, img2=img_children_ale)
_ = save(img_conj_ale, output_dir + "children_conj_adults_ale.nii.gz")

# %%
# Glass brain for adults only
p = plotting.plot_glass_brain(img_adults_z, display_mode="lyrz", colorbar=True)

# Table for adults only
t = reporting.get_clusters_table(img_adults_z, stat_threshold=0, min_distance=1000)
display(t)

# %%
# Glass brain for children vs. adults
p_sub = plotting.plot_glass_brain(
    img_sub,
    display_mode="lyrz",
    colorbar=True,
    vmax=5,
    plot_abs=False,
    symmetric_cbar=True,
)

# Table for children vs. adults
t_sub = reporting.get_clusters_table(
    img_sub, stat_threshold=0, min_distance=1000, two_sided=True
)
display(t_sub)

# %%
# Glass brain for conjunction
p_conj = plotting.plot_glass_brain(
    img_conj_z,
    display_mode="lyrz",
    colorbar=True,
    vmax=8,
)

# Table for children vs. adults
t_conj = reporting.get_clusters_table(
    img_conj_z, stat_threshold=0, min_distance=1000, two_sided=True
)
display(t_conj)
