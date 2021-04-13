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
import random
from math import sqrt
from os import makedirs, path
from re import sub
from shutil import copy
from sys import argv

import numpy as np
import pandas as pd
from nibabel import save
from nilearn import image, plotting, reporting
from nimare import correct, io, meta, utils
from scipy.stats import norm


# %%
# Define function to generate a new data set with k null studies added
def generate_null(
    text_file="foci.txt",
    space="ale_2mm",
    k_null=100,
    random_seed=None,
    output_dir="./",
):

    # Load NiMARE's gray matter template
    temp = utils.get_template(space=space, mask="gm")

    # Extract possible MNI coordinates for all gray matter voxels
    x, y, z = np.where(temp.get_fdata() == 1.0)
    within_mni = image.coord_transform(x=x, y=y, z=z, affine=temp.affine)
    within_mni = np.array(within_mni).transpose()

    # Read the original Sleuth file into a NiMARE data set
    dset = io.convert_sleuth_to_dataset(text_file, target=space)

    # Set random seed if requested
    if random_seed:
        random.seed(random_seed)

    # Resample numbers of subjects per experiment based on the original data
    nr_subjects_dset = [n[0] for n in dset.metadata["sample_sizes"]]
    nr_subjects_null = random.choices(nr_subjects_dset, k=k_null)

    # Do the same for the numbers of foci per experiment
    nr_foci_dset = dset.coordinates["study_id"].value_counts().tolist()
    nr_foci_null = random.choices(nr_foci_dset, k=k_null)

    # Create random foci
    idx_list = [
        random.sample(range(len(within_mni)), k=k_foci) for k_foci in nr_foci_null
    ]
    foci_null = [within_mni[idx] for idx in idx_list]

    # Create the destination Sleuth file
    makedirs(output_dir, exist_ok=True)
    text_file_basename = path.basename(text_file)
    null_file_basename = sub(
        pattern=".txt", repl="_plus_k" + str(k_null) + ".txt", string=text_file_basename
    )
    null_file = output_dir + "/" + null_file_basename
    copy(text_file, null_file)

    # Append all the null studies to this file
    f = open(null_file, mode="a")
    for i in range(k_null):
        f.write(
            "\n// nullstudy"
            + str(i + 1)
            + "\n// Subjects="
            + str(nr_subjects_null[i])
            + "\n"
        )
        np.savetxt(f, foci_null[i], fmt="%.3f", delimiter="\t")
    f.close()

    # Read the file and return it as a NiMARE data set
    dset_null = io.convert_sleuth_to_dataset(null_file, target=space)
    return dset_null


# %%
# Define function to compute the FSN for all clusters from a Sleuth file
def compute_fsn(
    text_file="foci.txt",
    space="ale_2mm",
    voxel_thresh=0.001,
    cluster_thresh=0.01,
    n_iters=1000,
    random_ale_seed=None,
    random_null_seed=None,
    output_dir="./",
):

    # Print what we're going to do
    print("\nCOMPUTING FSN FOR " + text_file + " (seed: " + str(random_null_seed) + ")")

    # Set random seed for original ALE if requested
    if random_ale_seed:
        np.random.seed(random_ale_seed)

    # Recreate the original ALE analysis
    ale = meta.cbma.ALE()
    corr = correct.FWECorrector(
        method="montecarlo", voxel_thresh=voxel_thresh, n_iters=n_iters
    )
    dset_orig = io.convert_sleuth_to_dataset(text_file=text_file, target=space)
    res_orig = ale.fit(dset_orig)
    cres_orig = corr.transform(res_orig)

    # Extract the original study IDs
    ids_orig = dset_orig.ids.tolist()

    # Create a new data set with a large number null studies added
    k_max = len(ids_orig) * 8
    dset_null = generate_null(
        text_file=text_file,
        space=space,
        k_null=k_max,
        random_seed=random_null_seed,
        output_dir=output_dir,
    )

    # Create thresholded cluster mask
    img_fsn = cres_orig.get_map("z_level-cluster_corr-FWE_method-montecarlo")
    cluster_thresh_z = norm.ppf(1 - cluster_thresh / 2)
    img_fsn = image.threshold_img(img_fsn, threshold=cluster_thresh_z)
    img_fsn = image.math_img("np.where(img > 0, 1, 0)", img=img_fsn)

    # Create cluster-thresholded z map
    img_z = cres_orig.get_map("z")
    img_z = image.math_img("img1 * img2", img1=img_fsn, img2=img_z)

    # Create cluster table where FSN will be added
    tab_fsn = reporting.get_clusters_table(img_z, stat_threshold=0, min_distance=1000)

    # Iteratively add null studies up to a certain maximum
    for k in range(1, k_max):

        # Print message
        print("Computing ALE for k = " + str(k) + " null studies added...")

        # Create a new data set
        ids_null = ["nullstudy" + str(x) + "-" for x in range(1, k + 1)]
        ids = ids_orig + ids_null
        dset_k = dset_null.slice(ids)

        # Compute the ALE
        res_k = res = ale.fit(dset_k)
        cres_k = corr.transform(result=res_k)

        # Create a thresholded cluster mask
        img_k = cres_k.get_map("z_level-cluster_corr-FWE_method-montecarlo")
        img_k = image.threshold_img(img_k, threshold=cluster_thresh_z)
        img_k = image.math_img("np.where(img > 0, 1, 0)", img=img_k)

        # Use this to update the per-voxel FSN
        # (i.e., increase the value of the voxel by 1 as long as it remains significant)
        count = str(k + 1)
        formula = "np.where(img_fsn + img_k == " + count + ", img_fsn + 1, img_fsn)"
        img_fsn = image.math_img(formula, img_fsn=img_fsn, img_k=img_k)

        # Quit as soon as there are no significant clusters left in the current map
        if not np.any(img_k.get_fdata()):
            print("No more significant voxels - terminating\n")
            break

    # Save the FSN map
    filename_img = path.basename(text_file).replace(".txt", "_fsn.nii.gz")
    save(img_fsn, filename=output_dir + "/" + filename_img)

    # Extract the FSN at the cluster peaks
    x, y, z = [np.array(tab_fsn[col]) for col in ["X", "Y", "Z"]]
    inv_affine = np.linalg.inv(img_z.affine)
    x, y, z = image.coord_transform(x=x, y=y, z=z, affine=inv_affine)
    x, y, z = [arr.astype("int") for arr in [x, y, z]]
    tab_fsn["FSN"] = img_fsn.get_fdata()[x, y, z]

    # Save the cluster table
    filename_tab = path.basename(text_file).replace(".txt", "_fsn.tsv")
    tab_fsn.to_csv(output_dir + "/" + filename_tab, sep="\t", index=False)

    return img_fsn, tab_fsn


# %%
# Define the Sleuth file names directly wihin the script
prefixes = ["all", "knowledge", "relatedness", "objects"]

# Or get the Sleuth file names for which to compute the FSN from the command line
prefixes = argv[1].split(",")

# List Sleuth files for which we want to perform an FSN analysis
text_files = ["../results/ale/" + prefix + ".txt" for prefix in prefixes]

# Create output directory based on these filenames
output_dirs = ["../results/fsn_full/" + prefix + "/" for prefix in prefixes]

# How many different filedrawers to compute for each text file?
nr_filedrawers = 10
filedrawers = ["filedrawer" + str(fd) for fd in range(nr_filedrawers)]

# Create a reproducible random seed for each filedrawer
random_master_seed = 1234
random.seed(random_master_seed)
random_null_seeds = random.sample(range(1000), k=nr_filedrawers)

# %%
# Use our function to compute multiple filedrawers for each text file
tabs_fsn, imgs_fsn = [
    [
        compute_fsn(
            text_file=text_file,
            space="ale_2mm",
            voxel_thresh=0.001,
            cluster_thresh=0.01,
            n_iters=1000,
            random_ale_seed=random_master_seed,
            random_null_seed=random_null_seed,
            output_dir=output_dir + filedrawer,
        )
        for random_null_seed, filedrawer in zip(random_null_seeds, filedrawers)
    ]
    for text_file, output_dir in zip(text_files, output_dirs)
]

# %%
# Read FSN tables
tab = [
    pd.read_csv(
        "../results/fsn_full/knowledge/filedrawer" + str(fd) + "/knowledge_fsn.tsv",
        delimiter="\t",
    )
    for fd in range(nr_filedrawers)
]
tab = pd.concat(tab)

# Compute summary statistics across filedrawers
agg = tab.groupby("Cluster ID")["FSN"].agg(["mean", "count", "std"])

# Compute confidence intervals
ci_level = 0.05
z_crit = abs(norm.ppf(ci_level / 2))
agg["se"] = [std / sqrt(count) for std, count in zip(agg["std"], agg["count"])]
agg["ci_lower"] = agg["mean"] - z_crit * agg["se"]
agg["ci_upper"] = agg["mean"] + z_crit * agg["se"]

# %%
# Plot mean FSN image across filedrawers
imgs_knowledge = [
    image.load_img(
        "../results/fsn_full/knowledge/filedrawer" + str(fd) + "/knowledge_fsn.nii.gz"
    )
    for fd in range(nr_filedrawers)
]
img_knowledge = image.mean_img(imgs_knowledge)
p = plotting.plot_glass_brain(None)
p.add_overlay(img_knowledge, colorbar=True, cmap="RdYlGn", vmin=0, vmax=100)
