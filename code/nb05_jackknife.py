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
from nb01_ale import write_foci, run_ale
import numpy as np
from os import makedirs, path
from nilearn import image, plotting
from nibabel import save

# %%
# Read table of experiments from ALE analysis
exps = pd.read_pickle("../results/exps.pickle")

# %%

# Create names of the Sleuth files to write
text_files = [
    "../results/jackknife/" + exp + "/" + exp + ".txt" for exp in exps["experiment"]
]

# For each experiment, write a Sleuth file leaving this one out
_ = [
    write_foci(
        text_file=text_file,
        df=exps,
        query="experiment != '" + exp + "'",
    )
    for text_file, exp in zip(text_files, exps["experiment"])
]

#%%

# Perform all of these ALEs
_ = [
    run_ale(
        text_file=text_file,
        voxel_thresh=0.001,
        cluster_thresh=0.01,
        n_iters=1000,
        output_dir=path.dirname(text_file),
    )
    for text_file in text_files
]

#%%

# Load all the thresholded maps we've created
img_files = [text_file.replace(".txt", "_z_thresh.nii.gz") for text_file in text_files]
imgs = [image.load_img(img_file) for img_file in img_files]

# Convert to binary cluster maps
masks = [image.math_img("np.where(img > 0, 1, 0)", img=img) for img in imgs]

# Compute the averaged jackknife map
img_mean = image.mean_img(masks)

# Save and plot
save(img_mean, filename="../results/jackknife/jackknife_mean.nii.gz")
p = plotting.plot_glass_brain(img_mean, colorbar=True)
