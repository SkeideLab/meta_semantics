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
# ![SkeideLab and MPI CBS logos](misc/header_logos.png)
#
# # Notebook #02: Subtraction Analysis
#
# *Created April 2021 by Alexander Enge* ([enge@cbs.mpg.de](mailto:enge@cbs.mpg.de))
#
# In this second notebook, we perform a couple of ALE subtraction analyses (also called contrast analyses; Laird et al., 2005, *Hum Brain Mapp*). These will inform us if and where there are reliable differences between two ALE images, e.g., for two different semantic task categories or for older as compared to younger children. The logic is to subtract the second ALE image from the first ALE image and compare the resulting difference scores to an empirical null distribution (derived from reshuffling the experiments into random groups and calculating new subtraction images under the null).
#
# We again start by loading the relevant packages.

# %%
from os import makedirs, path

import numpy as np
from IPython.display import display
from nibabel import save
from nilearn import glm, image, plotting, reporting
from nimare import io, meta
from numpy import random


# %% [markdown]
# Before starting with the actual subtraction analyses, let's define a helper function for statistical thresholding. Since no FWE correction method has been defined for subtraction analyses (yet), we use an uncorrected voxel-level threshold (usually $p<.001$) combined with a cluster-level extent threshold (in mm<sup>3</sup>). Note that we assume the voxel size to be $2\times2\times2$ mm<sup>3</sup> (the default in NiMARE).

# %%
# Define helper function for dual threshold based on voxel-p and cluster size (in mm3)
def dual_thresholding(
    img_z, voxel_thresh, cluster_size_mm3, two_sided=True, fname_out=None
):

    # If img_z is a file path, we first need to load the actual image
    img_z = image.load_img(img=img_z)

    # Check if the image is empty
    if np.all(img_z.get_fdata() == 0):
        print("THE IMAGE IS EMPTY! RETURNING THE ORIGINAL IMAGE.")
        return img_z

    # Convert desired cluster size to the corresponding number of voxels
    k = cluster_size_mm3 // 8

    # Actual thresholding
    img_z_thresh, thresh_z = glm.threshold_stats_img(
        stat_img=img_z,
        alpha=voxel_thresh,
        height_control="fpr",
        cluster_threshold=k,
        two_sided=two_sided,
    )

    # Print the thresholds that we've used
    print(
        "THRESHOLDING IMAGE AT Z > "
        + str(thresh_z)
        + " (P = "
        + str(voxel_thresh)
        + ") AND K > "
        + str(k)
        + " ("
        + str(cluster_size_mm3)
        + " mm3)"
    )

    # If requested, save the thresholded map
    if fname_out:
        save(img_z_thresh, filename=fname_out)

    return img_z_thresh


# %% [markdown]
# Now we can go on to perform the actual subtraction analyses. We again define a helper function for this so we can apply this two multiple Sleuth files with a single call (and also reuse it in later notebooks). We simply read two Sleuth files into NiMARE and let its `meta.cbma.ALESubtraction` do the rest as briefly described above. It outputs an unthresholded *z* score map which we then threshold using our helper function defined above.

# %%
# Define function for performing a single ALE subtraction analysis
def run_subtraction(
    text_file1,
    text_file2,
    voxel_thresh,
    cluster_size_mm3,
    random_seed,
    n_iters,
    output_dir,
):

    # Print the current analysis
    print(
        'SUBTRACTION ANALYSIS FOR "'
        + text_file1
        + '" VS. "'
        + text_file2
        + '" WITH '
        + str(n_iters)
        + " PERMUTATIONS"
    )

    # Set a random seed to make the results reproducible
    if random_seed:
        np.random.seed(random_seed)

    # Read Sleuth files
    dset1 = io.convert_sleuth_to_dataset(text_file=text_file1)
    dset2 = io.convert_sleuth_to_dataset(text_file=text_file2)

    # Actually perform subtraction analysis
    sub = meta.cbma.ALESubtraction(n_iters=n_iters, low_memory=False)
    sres = sub.fit(dset1, dset2)

    # Save the unthresholded z map
    img_z = sres.get_map("z_desc-group1MinusGroup2")
    makedirs(output_dir, exist_ok=True)
    name1 = path.basename(text_file1).replace(".txt", "")
    name2 = path.basename(text_file2).replace(".txt", "")
    prefix = output_dir + "/" + name1 + "_minus_" + name2
    save(img_z, filename=prefix + "_z.nii.gz")

    # Create and save thresholded z map
    dual_thresholding(
        img_z=img_z,
        voxel_thresh=voxel_thresh,
        cluster_size_mm3=cluster_size_mm3,
        two_sided=True,
        fname_out=prefix + "_z_thresh.nii.gz",
    )


# %% [markdown]
# We create a dictionary of Sleuth file names which we want to subtract from one another and supply each of these contrast to the function that we ahve just defined. Note that large numbers (≥ 10,000) of Monte Carlo iterations seem to be necessary to get stable results, but this requires very large amounts of memory. You may therefore want to decrease `n_iters` when trying out this code on a small local machine or on a cloud service.

# %%
if __name__ == "__main__":

    # Create dictionary for which subtraction analyses to run
    subtrs = dict(
        {
            "../results/ale/knowledge.txt": "../results/ale/nknowledge.txt",
            "../results/ale/relatedness.txt": "../results/ale/nrelatedness.txt",
            "../results/ale/objects.txt": "../results/ale/nobjects.txt",
            "../results/ale/older.txt": "../results/ale/younger.txt",
        }
    )

    # Use our function to perform the actual subtraction analyses
    for key, value in zip(subtrs.keys(), subtrs.values()):
        run_subtraction(
            text_file1=key,
            text_file2=value,
            voxel_thresh=0.001,
            cluster_size_mm3=200,
            random_seed=1234,
            n_iters=20000,
            output_dir="../results/subtraction",
        )

# %% [markdown]
# Let's look at the results for one of the subtraction analyses (here the analysis contrasting visual object category experiments against semantic knowledge experiments and semantic relatedness experiments). Note that both increases in ALE scores (i.e., objects > knowledge + relatedness) and decreases in ALE scores (i.e., knowledge + relatedness > objects) may be present within the same map, so we need to display both positive and negative *z* scores.

# %%
if __name__ == "__main__":

    # Glass brain example
    img = image.load_img(
        "../results/subtraction/objects_minus_nobjects_z_thresh.nii.gz"
    )
    p = plotting.plot_glass_brain(
        img,
        display_mode="lyrz",
        colorbar=True,
        vmax=5,
        plot_abs=False,
        symmetric_cbar=True,
    )

    # Cluster table example
    t = reporting.get_clusters_table(
        img, stat_threshold=0, min_distance=1000, two_sided=True
    )
    display(t)
