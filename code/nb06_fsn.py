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
# # Notebook #06: Fail-Safe N Analysis
#
# *Created April 2021 by Alexander Enge* ([enge@cbs.mpg.de](mailto:enge@cbs.mpg.de))
#
# This notebook describes another type of robustness check for our meta-analytic results. Here, we ask what happens with our inferences under the (not unlikely) assumption that there are additional studies with inconsistent findings that we've failed to include because they never got published (also called the "file drawer" effect). As for the jackknife analysis, we cannot directly test or correct for the presence of this bias. We can, however, perform simulations to estimate the impact that such a bias *would* have *if* it existed. The logic is simple and has been peformed for classical (i.e., non-neuroimaging) meta-analyses for decades (Rosenthal et al., 1979, *Psychol Bull*): Just create fictional studies with non-significant effect sizes and see how many of these can be added to your meta-analysis until the statistical significance of your effect size is overturned. If that number (called fail-safe N or FSN) exceeds any plausible guess of studies that *could* actually be in the file drawer, this would speak to the robustness of the meta-analytic results. Acar and colleagues (2018; *PLOS ONE*) have extended this logic to ALE-based meta-analysis of neuroimaging experiments: Just create new (fictional) experiments with inconsistent results and see how many of these can added to the original ALE until the significance of your clusters is overturned. Here we'll perform such an analysis for our four ALEs of interest (main analysis, knowledge, relatedness, and objects).
#
# As always, we start by loading the relevant packages.

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


# %% [markdown]
# Our FSN simulations will heavily rely on the generation of so called "null experiments," i.e., fictional experiments with their peaks randomly distributed across the brain. We'll start by writing a helper function for this. It takes as its input a "real" ALE data set (in the form of a Sleuth text file, see Notebook #1 and [this example](http://www.brainmap.org/ale/foci2.txt)). It then creates a desired number (`k_null`) of null experiments that are similar to the real experiments in terms of sample sizes and number of peak coordinates, but have the location of these peaks drawn randomly from all voxels within our gray matter template. For these experiments, we know that the null hypothesis of no spatial convergence is true, thus providing a testing ground for the file drawer effect.

# %%
# Define function to generate a new data set with k null studies added
def generate_null(
    text_file="peaks.txt",
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

    # Resample numbers of peaks per experiment based on the original data
    nr_peaks_dset = dset.coordinates["study_id"].value_counts().tolist()
    nr_peaks_null = random.choices(nr_peaks_dset, k=k_null)

    # Create random peaks
    idx_list = [
        random.sample(range(len(within_mni)), k=k_peaks) for k_peaks in nr_peaks_null
    ]
    peaks_null = [within_mni[idx] for idx in idx_list]

    # Copy original experiments to the destination Sleuth file
    makedirs(output_dir, exist_ok=True)
    text_file_basename = path.basename(text_file)
    null_file_basename = sub(
        pattern=".txt", repl="_plus_k" + str(k_null) + ".txt", string=text_file_basename
    )
    null_file = output_dir + "/" + null_file_basename
    copy(text_file, null_file)

    # Append all the null studies to the Sleuth file
    f = open(null_file, mode="a")
    for i in range(k_null):
        f.write(
            "\n// nullstudy"
            + str(i + 1)
            + "\n// Subjects="
            + str(nr_subjects_null[i])
            + "\n"
        )
        np.savetxt(f, peaks_null[i], fmt="%.3f", delimiter="\t")
    f.close()

    # Read the new Sleuth file and return it as a NiMARE data set
    dset_null = io.convert_sleuth_to_dataset(null_file, target=space)
    return dset_null


# %% [markdown]
# With this helper function, we can implement the actual FSN simulation. This function will be rather long-winded and complex, but the overall logic is simple: Take the Sleuth file from the original ALE analysis, generate a large number of null experimetns, and add them iteratively to the analysis. At each step, re-estimate the ALE and record which voxel have remained statistically significant and which have not. Based on this, each (initially singificant) voxel will be assigned an FSN value which is equal to the highest number of null experiments that we were able to add before the voxel failed to reach significant for the first time. Because these simulations are going to take a really (!) long time, we implement two stopping rules: We either stop if there are no more significant voxels left *or* if we've added a sufficiently large number of null studies for our purpose (e.g., five times times the number of original studies in the meta-analysis). 

# %%
# Define function to compute the FSN for all voxels from a Sleuth file
def compute_fsn(
    text_file="peaks.txt",
    space="ale_2mm",
    voxel_thresh=0.001,
    cluster_thresh=0.01,
    n_iters=1000,
    k_max_factor=5,
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
    k_max = len(ids_orig) * k_max_factor
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

    # Iteratively add null studies up to our pre-defined maximum
    for k in range(1, k_max):

        # Print message
        print("Computing ALE for k = " + str(k) + " null studies added...")

        # Create a new data set with k null studies added
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

        # Use this to update the per-voxel FSN - this is a bit hack-ish: On a voxel-by-
        # voxel basis, we increase the value by 1 if and only if the voxel has remained
        # significant. As soon as it has failed to reach significance once, we never
        # increase FSN any further. This is handeled by comparing the current FSN to
        # the current value of k.
        count = str(k + 1)
        formula = "np.where(img_fsn + img_k == " + count + ", img_fsn + 1, img_fsn)"
        img_fsn = image.math_img(formula, img_fsn=img_fsn, img_k=img_k)

        # Quit as soon as there are no significant clusters left in the current map
        if not np.any(img_k.get_fdata()):
            print("No more significant voxels - terminating\n")
            break

    # Save the FSN map that we've created in the loop
    filename_img = path.basename(text_file).replace(".txt", "_fsn.nii.gz")
    save(img_fsn, filename=output_dir + "/" + filename_img)

    # Extract the FSN values at the original cluster peaks
    tab_fsn = reporting.get_clusters_table(img_z, stat_threshold=0, min_distance=1000)
    inv_affine = np.linalg.inv(img_z.affine)
    x, y, z = [np.array(tab_fsn[col]) for col in ["X", "Y", "Z"]]
    x, y, z = image.coord_transform(x=x, y=y, z=z, affine=inv_affine)
    x, y, z = [arr.astype("int") for arr in [x, y, z]]
    tab_fsn["FSN"] = img_fsn.get_fdata()[x, y, z]

    # Save this cluster table with the new FSN column
    filename_tab = path.basename(text_file).replace(".txt", "_fsn.tsv")
    tab_fsn.to_csv(output_dir + "/" + filename_tab, sep="\t", index=False)

    return img_fsn, tab_fsn


# %% [markdown]
# Ideally, we want to perform all of this multiple times for different (random) filedrawers. Otherwise, the resulting FSN values would hinge a lot on the random patterns of these specific null experiments. However, doing all of these iterative simulations multiple times is extremly computationally expensive. We therefore wrote the next bit of the notebook in a way so that it can be run in parallel on our high performance computing (HPC) cluster. For this, we would call this notebook as a Python script from the command line and need to provide it with two additional parameters: The name of the original ALE analysis for which we want to compute the FSN and the number of different filedrawers we want to estimate (so we can always comput multiple filedrawers in parallel). If you don't happen to have access to an HPC and want to try the out the simulations directly withing the notebook, simply uncomment these two lines of code to define `prefixes` and `nr_filedrawers` locally. Note that here we're just setting things up so we can then compute a number of FSN maps for different file drawers for all of our ALE analyses.

# %%
# Get which FSN analyses to perform from the command line
prefixes = argv[1].split(",")

# Get number of filedrawers per analysis from the command line
nr_filedrawers = int(argv[2])

# # Or define them here for debugging
# prefixes = ["all", "knowledge", "relatedness", "objects"]
# nr_filedrawers = 5

# List the filenames of the Sleuth text files
text_files = ["../results/ale/" + prefix + ".txt" for prefix in prefixes]

# Create output directory names
output_dirs = ["../results/fsn/" + prefix + "/" for prefix in prefixes]

# Create random seeds for filedrawers
random_null_seeds = random.sample(range(1000), k=nr_filedrawers)
filedrawers = ["filedrawer" + str(seed) for seed in random_null_seeds]

# %% [markdown]
# The actual simulations are happening here: For each of the input text file (one for each original ALE analysis), we compute one FSN map for a number of different filedrawers (in our paper, we used ten). The results for each file drawer will be stored in separate folder which also indicates the random seed value (e.g., `166`) that was used to generate the null experiments (e.g., `results/fsn/all/filedrawer166`).

# %%
# Use our function to compute multiple filedrawers for each text file
_ = [
    [
        compute_fsn(
            text_file=text_file,
            space="ale_2mm",
            voxel_thresh=0.001,
            cluster_thresh=0.01,
            n_iters=1000,
            k_max_factor=5,
            random_ale_seed=1234,
            random_null_seed=random_null_seed,
            output_dir=output_dir + filedrawer,
        )
        for random_null_seed, filedrawer in zip(random_null_seeds, filedrawers)
    ]
    for text_file, output_dir in zip(text_files, output_dirs)
]

# %% [markdown]
# Once the simulation are done, we need to aggregate the results that we've obtained for each analysis across multiple file drawers. Remember that for each file drawer, we've already stored an FSN map as well as a table that contains the FSN value for each of the original cluster peaks. Here we just average those maps and perform some summary statistics on those tables, such as computing the mean FSN and its 95% confidence interval across the multiple (in our case, ten) file drawers.

# %%
# Compute mean FSN across filedrawers
for prefix in prefixes:

    # Read FSN maps from all filedrawers
    fnames_maps = glob(
        "../results/fsn/" + prefix + "/filedrawer*/" + prefix + "_fsn.nii.gz"
    )
    imgs_fsn = [image.load_img(fname) for fname in fnames_maps]

    # Average and save
    img_mean = image.mean_img(imgs_fsn)
    fname_img_mean = "../results/fsn/" + prefix + "/" + prefix + "_mean_fsn.nii.gz"
    save(img_mean, fname_img_mean)

    # Read FSN tables from all filedrawers
    fnames_tabs = glob(
        "../results/fsn/" + prefix + "/filedrawer*/" + prefix + "_fsn.tsv"
    )
    tabs_fsn = [pd.read_csv(fname, delimiter="\t") for fname in fnames_tabs]
    tab_fsn = pd.concat(tabs_fsn)

    # Compute summary statistics
    agg = tab_fsn.groupby("Cluster ID")["FSN"].agg(["mean", "count", "std"])

    # Compute confidence intervals
    ci_level = 0.05
    z_crit = abs(norm.ppf(ci_level / 2))
    agg["se"] = [std / sqrt(count) for std, count in zip(agg["std"], agg["count"])]
    agg["ci_lower"] = agg["mean"] - z_crit * agg["se"]
    agg["ci_upper"] = agg["mean"] + z_crit * agg["se"]

    # Save summary statistics
    fname_agg = "../results/fsn/" + prefix + "/" + prefix + "_mean_fsn.csv"
    agg.to_csv(fname_agg, float_format="%.3f")
