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
# ![SkeideLab and MPI CBS logos](misc/header_logos.png)
#
# # Notebook #05: Jackknife analysis
#
# *Created April 2021 by Alexander Enge* ([enge@cbs.mpg.de](mailto:enge@cbs.mpg.de))
#
# This notebook performs the first of two robustness checks that we're going to perform on our meta-analytic (ALE) results. This first one is called a "jackknife analysis" (or, also, a leave-one-out analysis). As the name says, we'll just repeat our original meta-analysis multiple times, each time leaving out a different one of original experiments that went into it. By examining if and how the results change across these simulations, we'll get an idea of how much our meta-analytic clusters hinge on any individual study (which may or may not report spurious peaks). Note that this involves running a large number of (synthetic) ALE analyses and may therefore take multiple hours, especially if only few cores are available on your local machine or cloud server.
#
# As usual, we start by loading all the packages we'll need in this notebook.

# %%
from os import makedirs

import numpy as np
from IPython.display import display
from nibabel import save
from nilearn import image, plotting, reporting
from nimare import correct, io, meta
from scipy.stats import norm


# %% [markdown]
# Because we want to perform the jackknife analysis for multiple ALEs (i.e., the main analysis and the three task category-specific analyses), let's again define a helper function. This takes as its input a Sleuth text file (the same that we've created in Notebook #01 to run the original ALE analysis) plus a couple of additional parameters that will be used for all ALE simulations. These, of course, should be the same as for the original analysis for the jackknife results to be meaningful.
#
# The logic of our function is to read the Sleuth file and re-create the original ALE analysis (with all experiments inlcuded). Then, it loops over all those experiments and, at each iteration, drops the current one from the sample and re-estimates the ALE. The resulting meta-analytic map is converted into a binary mask telling us which voxels remained statistically significant ($0$ = not signficant, $1$ = significant). Once we've done this for all experiments, these images are averaged into a "jackknife map," simply showing us for each cluster the percentage of simulations in which it has remained significant. This can be seen as an indicator for the robustness of the cluster against spurious results in the meta-analytic sample.

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


# %% [markdown]
# So far we've only defined the jackknife function; now let's apply it to our ALE analyses. We just need to list the Sleuth text file names and provide our default ALE thresholds and settings. The function will run for a couple of hours and return, for each input Sleuth file, an averaged jackknife map.

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

# %% [markdown]
# Let's examine one of these maps, here for our main analysis containing all semantic knowledge experiments. Note that most clusters have a jackknife reliability of $1.0$, indicating strong robustness against the deletion of individual experiments. Only for two of the posterior clusters in the right hemisphere, the robustness is slightly reduced (but still high with approx. $0.85$).

# %%
# Glass brain example
img_jk_all = image.load_img("../results/jackknife/all/mean_jk.nii.gz")
p = plotting.plot_glass_brain(None, display_mode="lyrz", colorbar=True)
p.add_overlay(img_jk_all, colorbar=True, cmap="RdYlGn", vmin=0, vmax=1)

# Cluster table example
t = reporting.get_clusters_table(img_jk_all, stat_threshold=0, min_distance=1000)
display(t)
