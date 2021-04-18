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
import logging
import random
from math import sqrt
from os import makedirs, path
from re import sub
from shutil import copy
from sys import argv

import numpy as np
import pandas as pd
from nibabel import save
from nilearn import image, regions, reporting
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
    dset = io.convert_sleuth_to_dataset(text_file, target="ale_2mm")

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

    # Write all the null studies to this file
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
    print(
        "\nCOMPUTING FSN FOR ALL CLUSTERS IN "
        + text_file
        + " (random seed: "
        + str(random_null_seed)
        + ")\n"
    )

    # Set random seed for original ALE if requested
    if random_ale_seed:
        np.random.seed(random_ale_seed)

    # Specify ALE and FWE transformers
    ale = meta.cbma.ALE()
    corr = correct.FWECorrector(
        method="montecarlo", voxel_thresh=voxel_thresh, n_iters=n_iters
    )

    # Perform ALE analysis for the original Sleuth file
    print("Recreating original ALE analysis (k = 0)")
    dset_orig = io.convert_sleuth_to_dataset(text_file=text_file, target=space)
    res_orig = ale.fit(dset_orig)
    cres_orig = corr.transform(res_orig)

    # Store a list of the study IDs of the original studies
    ids_orig = dset_orig.ids.tolist()

    # Extract thresholded cluster image
    cluster_thresh_z = norm.ppf(1 - cluster_thresh / 2)
    img_orig = cres_orig.get_map("z_level-cluster_corr-FWE_method-montecarlo")
    img_orig = image.threshold_img(img_orig, threshold=cluster_thresh_z)

    # Create a table of the significant clusters
    cl_table = reporting.get_clusters_table(
        stat_img=img_orig, stat_threshold=cluster_thresh_z, min_distance=np.inf
    )
    cl_table[["FSN sampling", "FSN"]] = np.nan

    # Create a separate map for each significant cluster
    imgs_orig, _ = regions.connected_regions(
        img_orig, min_region_size=0, extract_type="connected_components"
    )

    # Determine the maximal number of null studies to add (original studies x 15)
    k_max = len(dset_orig.ids) * 15

    # Create a new data set with null studies added
    dset_null = generate_null(
        text_file=text_file,
        space=space,
        k_null=k_max,
        random_seed=random_null_seed,
        output_dir=output_dir,
    )

    # Create an empty cache to store images that we've already computed
    cache_imgs = dict()

    # Create an empty list where will store maps with the FSN for each cluster
    imgs_fsn = list()

    # Perform FSN sampling algorithm for each cluster
    nr_cls = len(cl_table)
    for cl in range(nr_cls):

        # Print what we're going to do
        cl_id = cl_table["Cluster ID"][cl]
        print("\nCOMPUTING FSN FOR CLUSTER #" + str(cl_id) + " OF " + str(nr_cls) + ":")

        # Extract the map containing only the current cluster
        img_cluster = image.index_img(imgs_orig, index=cl)

        # Create lists to keep track of the sampleing process:
        # (1) How many null studies did we use at this step?
        ks = np.array([0], dtype="int")
        # (2) Was our cluster at this step significant or not?
        sigs = np.array([True], dtype="bool")

        # Start sampling different values for k
        while True:

            # For the first iteration, k = k_max
            if len(ks) == 1:
                k = k_max
            # If not, it is set to the midpoint between highest number of k where cl was
            # significant and the lowest number of k where cl was not significant
            else:
                k = np.mean([ks[sigs].max(), ks[~sigs].min()], dtype="int")

            # We've found our FSN as soon as we've encountered the same value twice
            if k in ks:
                break

            # See if the necessary image is alread in the cache
            key_k = "k" + str(k)
            if key_k in cache_imgs:

                # If so, just load the image from the cache
                print("Loading image from cache for k = " + str(k) + "... ", end="")
                img_k = cache_imgs[key_k]

            else:

                # If not, we compute a new ALE with k null studies added
                print("Computing ALE for k = " + str(k) + "... ", end="")

                # Create a new data set
                ids_null = ["nullstudy" + str(x + 1) + "-" for x in range(k)]
                ids = ids_orig + ids_null
                dset_k = dset_null.slice(ids)

                # Compute the ALE
                res_k = res = ale.fit(dset_k)
                cres_k = corr.transform(result=res_k)

                # Write the image to the cache
                img_k = cres_k.get_map("z_level-cluster_corr-FWE_method-montecarlo")
                cache_imgs[key_k] = img_k

            # Check if the cluster has remained significant
            img_k = image.threshold_img(img_k, threshold=cluster_thresh_z)
            img_check = image.math_img("img1 * img2", img1=img_cluster, img2=img_k)
            cluster_significant = np.any(img_check.get_fdata())
            if cluster_significant:
                print("significant.")
            else:
                print("not significant.")

            # Add the current iterations to the lists and continue
            ks = np.append(ks, k)
            sigs = np.append(sigs, cluster_significant)

        # Once we've found FSN, we create a map showing the FSN for the current cluster
        print("DONE SAMPLING FOR CLUSTER #" + str(cl) + ": FSN = " + str(k))
        formula = "np.where(img != 0, " + str(k) + ", 0)"
        img_cluster = image.math_img(formula=formula, img=img_cluster)
        imgs_fsn.append(img_cluster)

        # And, finally, we also add the FSN to the cluster table
        cl_table["FSN sampling"][cl] = ks.astype(object)
        cl_table["FSN"][cl] = k

    # Save and return the cluster table
    makedirs(output_dir, exist_ok=True)
    prefix = sub(".txt", "", path.basename(text_file))
    cl_table.to_csv(output_dir + "/" + prefix + "_cluster_fsn.csv", index=False)

    # Compute and save the comple FSN map
    img_fsn = image.mean_img(imgs_fsn)
    formula = "img * " + str(nr_cls)
    img_fsn = image.math_img(formula=formula, img=img_fsn)
    save(img_fsn, output_dir + "/" + prefix + "_cluster_fsn.nii.gz")

    return cl_table, img_fsn


# %%
# # Define the Sleuth file names directly wihin the script
# prefixes = ["all", "knowledge", "relatedness", "objects"]

# Get the Sleuth file names for which to compute the FSN from the command line
prefixes = argv[1].split(",")

# List Sleuth files for which we want to perform an FSN analysis
text_files = ["../results/ale/" + prefix + ".txt" for prefix in prefixes]

# Create output directory based on these filenames
output_dirs = ["../results/fsn/" + prefix + "/" for prefix in prefixes]

# How many different filedrawers to compute for each text file?
nr_filedrawers = 10
filedrawers = ["filedrawer" + str(fd) for fd in range(nr_filedrawers)]

# Create a reproducible random seed for each filedrawer
random_master_seed = 1234
random.seed(random_master_seed)
random_null_seeds = random.sample(range(0, 1000), k=nr_filedrawers)

# Use our function to compute multiple filedrawers for each text file
_ = [
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
        "../results/fsn/knowledge/filedrawer" + str(fd) + "/knowledge_cluster_fsn.csv",
    )
    for fd in range(10)
]
tab = pd.concat(tab)

# Compute summary statistics across filedrawers
agg = tab.groupby(["X", "Y", "Z"])["FSN"].agg(["mean", "count", "std"])

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
        "../results/fsn/knowledge/filedrawer" + str(fd) + "/knowledge_cluster_fsn.nii.gz"
    )
    for fd in range(10)
]
img_knowledge = image.mean_img(imgs_knowledge)
p = plotting.plot_glass_brain(None)
p.add_overlay(img_knowledge, colorbar=True, cmap="RdYlGn", vmin=0, vmax=100)
