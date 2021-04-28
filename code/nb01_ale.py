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
# # Notebook #01: Activation Likelihood Estimation
#
# *Created April 2021 by Alexander Enge* ([enge@cbs.mpg.de](mailto:enge@cbs.mpg.de))
#
# This first notebook computes multiple coordinate-based meta-analyses using the activation likelihood estimation (ALE) algorithm (Eickhoff et al., 2009, *Hum Brain Mapp*; 2012, *NeuroImage*; Turkeltaub et al., 2002, *NeuroImage*). Based on peak fMRI coordinates reported in a common standard space, ALE estimates where in the brain there is spatial convergence of activation across multiple experiments on a given topic (here: semantic cognition in children). We have listed all of these experiments together with some descriptive information (e.g., sample sizes, age of the children) in an `included.csv` spreadsheet. The peak coordinates for each of these experiments are stored in separate `.csv` files. To compute the ALE, these need to be converted into a common standard space and combined into single text file (called a "Sleuth" file; see [here](http://www.brainmap.org/ale/foci2.txt) for an example). This can then be fed into the ALE algorithm as implemented in the [NiMARE package](https://nimare.readthedocs.io). We do all of this once for our whole meta-analytic sample (including all semantic knowledge experiments) and once for each of three subgroups of experiments (semantic knowledge experiments, semantic relatedness experiments, and visual semantic object category experiments).
#
# We start by loading the relevant packages.

# %%
from os import makedirs, path

import numpy as np
import pandas as pd
from IPython.display import display
from nibabel import save
from nilearn import image, plotting, reporting
from nimare import correct, io, meta, utils
from scipy.stats import norm

# %% [markdown]
# Next, we read the spreadsheet file with the description information about all experiments. It contains one row for each experiment and one column for each variable (sample size `n`, `age_mean` of the children etc.), not all of which are important here. Note that the `if __name__ == "__main__"` statement doesn't do any work here—it just needs to be there because we want to reuse the functions that we're defining here in some of the later notebooks.

# %%
if __name__ == "__main__":

    # Read spreadsheet with included experiments
    exps = pd.read_csv(
        "../data/literature_search/included.csv",
        na_filter=False,
        converters={"age_mean": float, "age_min": float, "age_max": float},
    )

    # Let's take a look
    display(exps)

# %% [markdown]
# Note that for two of these experiments, the `age_mean` has not been reported in the original article. Because we want to look at age-related changes later on, we need to fill in these missing values. As a proxy we simply use the midpoint of the age range (which has been reported for both of them). We also compute the median of `age_mean` across all experiments which we will use later on to perform a median split analysis (i.e., older vs. younger children).

# %%
if __name__ == "__main__":

    # Fill in mean age if missing (using the midpoint of min and max)
    exps["age_mean"] = [
        np.mean([age_min, age_max]) if np.isnan(age_mean) else age_mean
        for age_mean, age_min, age_max in zip(
            exps["age_mean"], exps["age_min"], exps["age_max"]
        )
    ]

    # Compute median of mean ages (for median split analysis)
    age_md = exps["age_mean"].median()

# %% [markdown]
# The next step is to get the actual peak coordinates for all experiments. We've already extracted these from the original papers and stored them into separte `.csv` files. From their, we can directly read them into a new column of our DataFrame. They are stored as $k\times3$ or $k\times4$ NumPy arrays (where $k$ is the number of reported peaks for this experiment). The first three columns contain the x, y, and z coordinates and the fourth column (if available) contains the peak test statistic (a *t* score or *z* score). Since some of the experiments reported the coordinates in Talairach space, we also need to convert those into our common standard MNI space using the *icbm2tal* transform (Lancaster et al., 2007, *Hum Brain Mapp*).

# %%
if __name__ == "__main__":

    # Read peak coordinates from .csv files
    exps["csv"] = "../data/peaks/" + exps["experiment"] + ".csv"
    exps["peaks"] = [
        np.genfromtxt(csv, delimiter=",", skip_header=1) for csv in exps["csv"]
    ]

    # Make sure all peaks are stored as 2D NumPy arrays
    exps["peaks"] = [
        np.expand_dims(peaks, axis=0) if np.ndim(peaks) != 2 else peaks
        for peaks in exps["peaks"]
    ]

    # Convert coordinates from Talairach to MNI space if necessary
    exps["peaks_mni"] = [
        utils.tal2mni(peaks[:, 0:3]) if peaks_space == "TAL" else peaks[:, 0:3]
        for peaks, peaks_space in zip(exps["peaks"], exps["peaks_space"])
    ]

    # Backup the DataFrame for reuse in other notebooks
    makedirs("../results/", exist_ok=True)
    exps.to_json("../results/exps.json")

# %% [markdown]
# With this, we can go on to write the Sleuth text files on which the actual ALEs will be performed. Each of these text files needs to contain the coordinates from all the relevant experiments. We define a helper function that extracts these from our DataFrame and writes them into a new text file that has the required Sleuth format.

# %%
# Define function to write a certain subset of the experiments to a Sleuth text file
def write_peaks_to_sleuth(text_file, df, query):

    makedirs(path.dirname(text_file), exist_ok=True)
    f = open(file=text_file, mode="w")
    f.write("// Reference=MNI\n")
    f.close()
    f = open(file=text_file, mode="a")
    df_sub = df.query(query)
    for experiment, n, peaks_mni in zip(
        df_sub["experiment"], df_sub["n"], df_sub["peaks_mni"]
    ):
        f.write("// " + experiment + "\n// Subjects=" + str(n) + "\n")
        np.savetxt(f, peaks_mni, fmt="%1.3f", delimiter="\t")
        f.write("\n")
    f.close()


# %% [markdown]
# We can than create a dictionary of output file names and queries to write the Sleuths files for all the ALEs we want to compute: One containing all experiments, one for each individual semantic task category (plus one for the two inverse categories), and one for older and younger children, respectively. Once we apply our function to this dictionary, the Sleuth files should show up in the `results/ale/` directory.

# %%
if __name__ == "__main__":

    # Create dictionary for which ALE analyses to run
    ales = dict(
        {
            "../results/ale/all.txt": "experiment == experiment",
            "../results/ale/knowledge.txt": 'task_type == "knowledge"',
            "../results/ale/nknowledge.txt": 'task_type != "knowledge"',
            "../results/ale/relatedness.txt": 'task_type == "relatedness"',
            "../results/ale/nrelatedness.txt": 'task_type != "relatedness"',
            "../results/ale/objects.txt": 'task_type == "objects"',
            "../results/ale/nobjects.txt": 'task_type != "objects"',
            "../results/ale/older.txt": "age_mean > @age_md",
            "../results/ale/younger.txt": "age_mean <= @age_md",
        }
    )

    # Use our function to write the Sleuth files
    for key, value in zip(ales.keys(), ales.values()):
        write_peaks_to_sleuth(text_file=key, df=exps, query=value)


# %% [markdown]
# We are now ready to perform the actual ALE analyses with NiMARE. We write a custom function which takes a single Sleuth text file as its input and (a) calculates the ALE map, (b) corrects for multiple comparisons using a Monte Carlo-based FWE correction, and (c) stores the cluster level-thresholded maps into the output directory. We then apply this function to all the Sleuth files we have created in the previous step. Note that on a machine or cloud server with a limited number of CPUs (such as Binder), each ALE will take a couple of minutes due to the FWE correction.

# %%
# Define function for performing a single ALE analysis with FWE correction
def run_ale(text_file, voxel_thresh, cluster_thresh, random_seed, n_iters, output_dir):

    # Let's show the user what we are doing
    print("ALE ANALYSIS FOR '" + text_file + "' WITH " + str(n_iters) + " PERMUTATIONS")

    # Set a random seed to make the results reproducible
    if random_seed:
        np.random.seed(random_seed)

    # Perform the ALE
    dset = io.convert_sleuth_to_dataset(text_file=text_file, target="ale_2mm")
    ale = meta.cbma.ALE()
    res = ale.fit(dset)

    # FWE correction for multiple comparisons
    corr = correct.FWECorrector(
        method="montecarlo", voxel_thresh=voxel_thresh, n_iters=n_iters
    )
    cres = corr.transform(result=res)

    # Save unthresholded maps to the ouput directory
    prefix = path.basename(text_file).replace(".txt", "")
    res.save_maps(output_dir=output_dir, prefix=prefix)
    cres.save_maps(output_dir=output_dir, prefix=prefix)

    # Create cluster-level thresholded z and ALE maps
    img_clust = cres.get_map("z_level-cluster_corr-FWE_method-montecarlo")
    img_z = cres.get_map("z")
    img_ale = cres.get_map("stat")
    cluster_thresh_z = norm.ppf(1 - cluster_thresh / 2)
    img_clust_thresh = image.threshold_img(img=img_clust, threshold=cluster_thresh_z)
    img_mask = image.math_img("np.where(img > 0, 1, 0)", img=img_clust_thresh)
    img_z_thresh = image.math_img("img1 * img2", img1=img_mask, img2=img_z)
    img_ale_thresh = image.math_ƒimg("img1 * img2", img1=img_mask, img2=img_ale)

    # Save thresholded maps to the output directory
    save(img=img_z_thresh, filename=output_dir + "/" + prefix + "_z_thresh.nii.gz")
    save(img=img_ale_thresh, filename=output_dir + "/" + prefix + "_stat_thresh.nii.gz")


if __name__ == "__main__":

    # Apply our function to all the Sleuth files
    for key in ales.keys():
        run_ale(
            text_file=key,
            voxel_thresh=0.001,
            cluster_thresh=0.01,
            random_seed=1234,
            n_iters=1000,
            output_dir="../results/ale/",
        )


# %% [markdown]
# Finally, let's look at some exemplary results by plotting the (cluster-level FWE-corrected) *z* score map from the main analysis (including all semantic experiments). We also print a table of the corresponding cluster statistics.

# %%
if __name__ == "__main__":

    # Glass brain example
    img = image.load_img("../results/ale/all_z_thresh.nii.gz")
    p = plotting.plot_glass_brain(img, display_mode="lyrz", colorbar=True)

    # Cluster table example
    t = reporting.get_clusters_table(img, stat_threshold=0, min_distance=1000)
    display(t)
