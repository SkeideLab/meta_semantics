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

# %%
from os import makedirs

import numpy as np
from IPython.display import display
from nibabel import save
from nilearn import image, plotting, reporting
from nimare import correct, io, meta
from scipy.stats import norm


# %%
# Define function to perform a single jackknife analysis
def compute_jackknife(
    text_file="foci.txt",
    space="ale_2mm",
    voxel_thresh=0.001,
    cluster_thresh=0.01,
    n_iters=1000,
    random_seed=None,
    output_dir="./",
):

    # Set random seeds if requested
    if random_seed:
        np.random.seed(random_seed)

    # Create NiMARE data set from the Sleuth file
    dset_orig = io.convert_sleuth_to_dataset(text_file)
    study_ids = dset_orig.ids

    # Specify ALE and FWE transformers
    ale = meta.cbma.ALE()
    corr = correct.FWECorrector(
        method="montecarlo", voxel_thresh=voxel_thresh, n_iters=n_iters
    )

    # Create output folder
    _ = makedirs(output_dir, exist_ok=True)

    # Create empty list to store the jackknife'd cluster images
    imgs_jk = []

    for study_id in study_ids:

        # Create new data set with the current study removed
        study_ids_jk = study_ids[study_ids != study_id]

        # Fit the jackknife'd ALE
        dset_jk = dset_orig.slice(study_ids_jk)
        res_jk = ale.fit(dset_jk)
        cres_jk = corr.transform(res_jk)

        # Create and save the thresholded cluster mask
        img_jk = cres_jk.get_map("z_level-cluster_corr-FWE_method-montecarlo")
        cluster_thresh_z = norm.ppf(1 - cluster_thresh / 2)
        formula = "np.where(img > " + str(cluster_thresh_z) + ", 1, 0)"
        img_jk = image.math_img(formula, img=img_jk)

        # Save to the output folder and to our list
        study_id_short = study_id.replace("-", "")
        save(img_jk, filename=output_dir + "/jk_" + study_id_short + ".nii.gz")
        imgs_jk.append(img_jk)

    # Create and save averaged jackknife image
    img_mean = image.mean_img(imgs_jk)
    save(img_mean, filename=output_dir + "/mean_jk.nii.gz")

    return img_mean


# %%
# List the Sleuth files for which to run a jackknife analysis
prefixes = ["all", "knowledge", "relatedness", "objects"]
text_files = ["../results/ale/" + prefix + ".txt" for prefix in prefixes]

# Create output directory names
output_dirs = ["../results/jackknife/" + prefix for prefix in prefixes]

# Apply the jackknife function
jks = [
    compute_jackknife(
        text_file=text_file,
        space="ale_2mm",
        voxel_thresh=0.001,
        cluster_thresh=0.01,
        n_iters=1000,
        random_seed=1234,
        output_dir=output_dir,
    )
    for text_file, output_dir in zip(text_files, output_dirs)
]

# %%
# Glass brain example
img_jk_all = image.load_img("../results/jackknife/all/mean_jk.nii.gz")
p = plotting.plot_glass_brain(None, display_mode="lyrz", colorbar=True)
p.add_overlay(img_jk_all, colorbar=True, cmap="RdYlGn", vmin=0, vmax=1)

# Cluster table example
t = reporting.get_clusters_table(img_jk_all, stat_threshold=0, min_distance=1000)
display(t)
