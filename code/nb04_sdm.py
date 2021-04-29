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
# # Notebook #04: Seed-Based *d* Mapping
#
# *Created April 2021 by Alexander Enge* ([enge@cbs.mpg.de](mailto:enge@cbs.mpg.de))
#
# In this notebook, we go back to the child-specific meta-analysis of fMRI studies of semantic cognition (see Notebook #01). This time, however, we're using a different meta-analytic algorithm called seed-based *d* mapping or SDM (Albajes-Eizagirre et al., 2019, *NeuroImage*). Unlike ALE, the SDM algorithm doesn't treat the peak coordinates as binary (i.e., $1$ = peak, $0$ = no peak). Instead, it uses the actual test statistics that are often provided for each peak coordinate in the original papers (usually in the form of *t* scores or *z* scores). Based on these, SDM runs a couply of complicated procedures under the hood to estimate effect size maps based on the known and unknown statistical values of all voxels. These experiment-specific effect size maps can then be combined meta-analytically in much the same way as one would combine effect sizes for behavioral or clinical outcomes in conventional (i.e., non-neuroimaging) meta-analyses. All of this is implemented conveniently a single toolbox which can be downloaded from the [SMD website](https://www.sdmproject.com). Since not only has a GUI but also a command line interface, we can directly call it from within this Python notebook.
#
# Before doing so, however, we need to go through some additional steps to retrieve the correct test statistic for all the peak coordinates. We start by loading some packages.

# %%
from glob import glob
from multiprocessing import cpu_count
from os import makedirs
from re import sub
from subprocess import run

import numpy as np
import pandas as pd
from IPython.display import display
from nilearn import image, plotting, reporting
from scipy import stats

from nb02_subtraction import dual_thresholding

# %% [markdown]
# Back in Notebook #01, we created an `exps.json` file that contained all the relevant information about our experiments. These include not only the peak coordinates (stored as NumPy arrays), but also some descriptive information about the sample of each experiments (e.g., sample size `n`, `age_mean` of the children) and the type of statistical threshold used by the original authors (e.g., $p<.001$ at the voxel level). Let's get this file back into a DataFrame.

# %%
# Read table of experiments from ALE analysis
exps = pd.read_json("../results/exps.json")
exps["peaks"] = [np.array(peaks, dtype="float") for peaks in exps["peaks"]]

# %% [markdown]
# As introduced above, SDM will use the test statistic for each peak coordinate to estimate effect size maps. These need to be provided in the common metric of *t* scores. In the original papers, these are more frequently reqorted as *z* scores, which can easily be converted into *t* scores by knowing the sample size of the experiment (because $df=n_{children}-1$). When neither *t* scores nor *z* scores are available for the peaks from a given experiment, we write the letter `p` by convention of SDM. We also chop of some unrealistically high *t* scores ($>50$) as these are likely to result from errors in statistical reporting. Note that this only was the case for ~5% of all peaks and did not alter the results in any meaningful way.

# %%
# Extract test statistics of individal peaks
exps["tstats"] = [
    
    # If there are t scores, we can use them directly
    peaks[:, 3] if peaks_stat == "tstat"
    
    # If there are z scores, we convert them to t scores
    else (
        stats.t.ppf(stats.norm.cdf(peaks[:, 3]), df=n - 1)
        if peaks_stat == "zstat"
        
        # If neither of these, write NaNs
        else np.full(peaks.shape[0], np.nan)
    )
    for peaks, peaks_stat, n in zip(exps["peaks"], exps["peaks_stat"], exps["n"])
]

# Replace missing and unrealistically high t scores
exps["tstats_corr"] = [
    np.where(np.isnan(tstats), "p", np.where(tstats > 50, 50, tstats))
    for tstats in exps["tstats"]
]

# How many of these do we have (absolute number and percentage)?
tstats_expl = np.array(exps["tstats"].explode(), dtype="float")
print(sum(np.isnan(tstats_expl)), sum(np.isnan(tstats_expl)) / tstats_expl.size)
print(sum(tstats_expl > 50), sum(tstats_expl > 50) / tstats_expl.size)

# Add new test statistics back to the peaks
exps["peaks_sdm"] = [
    np.c_[peaks, tstats_corr]
    for peaks, tstats_corr in zip(exps["peaks_mni"], exps["tstats_corr"])
]

# %% [markdown]
# Unlike ALE, where all experiments are fed into the algorithm from a single text file (the "Sleuth" file), SDM wants one text file for every experiment. This file contains one row per peak with its x, y, and z coordinate as well as its *t* score (or the letter *p* if no *t* score is available). The file name of this `.txt` file contains the name of the experiment and the standard space in which the peak coordinates are reported. Note that in our case all coordinates are already in a common MNI space (as per Notebook #01), even though some of the experiments originally used the Talairach space.

# %%
# Write the peaks of each experiment to a text file
makedirs("../results/sdm/", exist_ok=True)
_ = [
    np.savetxt(
        fname="../results/sdm/" + exp + ".other_mni.txt",
        X=peaks,
        fmt="%s",
        delimiter=",",
    )
    for exp, peaks in zip(exps["experiment"], exps["peaks_sdm"])
]

# %% [markdown]
# Besides the individual peak statistics, we also need to take care of fact that the experiments used different statistical thresholds and different correction procedures for the multiple comparisons problem. SDM will use this information when guessing the effect sizes for those peak coordinates for which no *t* score is available. Thus, we need to provide the correct *t* score threshold (called `t_thr`) for each experiment in the so-called "study table" (just another text file besides all the experiment-specific ones). This table will soon also contain some additional information about all experiments (more on this below). For now, let's focus on inferring `t_thr` for each experiment. Since not all of them explicitly report their voxel-level threshold as a *t* score, we again need to infer it from a *z* score threshold or, most commonly, from a *p* value threshold (usually $p<.001$ by convetion). For a few papers, neither a *t*, *z*, or *p* value threshold is provided; in these cases we simply assume that the threshold was identical to the lowest *t* score that was observed for the peaks from this experiment (which we just computed above). Note that this is being conservative since the actual *t* score threshold in the original paper may have been lower.

# %%
# Convert some columns from str to float
cols_thresh = ["thresh_vox_z", "thresh_vox_t", "thresh_vox_p"]
exps[cols_thresh] = exps[cols_thresh].apply(pd.to_numeric, errors="coerce")

# Determine a t score threshold for each experiment
exps["t_thr"] = [
    
    # If there is a t score threshold in the paper, we can use it directly
    t if not np.isnan(t)
    
    # If there is a z score threshold in the paper, we convert it to a t score
    else (
        stats.t.ppf(stats.norm.cdf(z), df=n - 1)
        if not np.isnan(z)
        
        # If there is a p value threshold in the paper, we convert it to a t score
        else (
            abs(stats.t.ppf(p, df=n - 1))
            if not np.isnan(p)
            
            # If none of these, use the lowest observed (significant) t score
            else pd.to_numeric(tstats).min()
        )
    )
    for t, z, p, n, tstats in zip(
        exps["thresh_vox_t"],
        exps["thresh_vox_z"],
        exps["thresh_vox_p"],
        exps["n"],
        exps["tstats"],
    )
]

# Backup DataFrame for reuse in other notebooks
exps.to_json("../results/exps.json")

# %% [markdown]
# As briefly touched upon, we also want our study table to contain some additional information besides the *t* score threshold. Three of them are required by SDM, namely the name of the experiment (in a column called `study`), the sample size (in a column called `n1`), and the type of voxel-level threshold (corrected or uncorrected; in a column called `threshold`). However, the effect size-based approach of SDM has the nice side effect that also allows us to include linear covariates into our meta-analytic model–something that is impossible in ALE. To do so, we write a small helper function that converts one column of our DataFrame (assumed to be in string format) to numerical values. In our order of choice, the first string gets a value of `0`, the second string a value of `1`, and so on. Note that this is a very crude way of putting categorical predictors into a linear model and their may be more elegant ones (such as contrast coding; Schad et al., 2020, *J Mem Lang*). However, we'll stick to this simply way because SDM only allows us to include up to four columns of covariates anyway.

# %%
# Create a copy of our DataFrame and rename some columns
exps_sdm = exps.rename(columns=({"experiment": "study", "n": "n1"}))


# Define function to convert string variables to integers (as dummies for categories)
def str_via_cat_to_int(series_in, categories):


    # Convert strings to category codes (in the provided order), starting at 1
    series_out = np.array(
        pd.Categorical(series_in).set_categories(new_categories=categories).codes
    )
    
    # Add another category code for any leftover categories
    series_out[series_out == -1] = series_out.max() + 1
    return series_out

# %% [markdown]
# We use this helper function to convert some of the columns that are already in our DataFrame into a numerical format that can be used by SDM. Covariates we may be interested include the semantic task category (knowledge, relatdness, or objects), the modality of stimulus presentation (e.g., visual, auditory), the modality of childrens' response (e.g., manual, covert speech), and the statistical software package used by the original authors for their analysis (SPM, FSL, or something else). We also include the mean sample age of the experiment and its square as additional (potential) covariates. Those of course are already in a numerical format and thus don't need to be converted.

# %%
# Apply this function to convert some columns to integers
cols_convert = ["task_type", "modality_pres", "modality_resp", "software"]
exps_sdm[cols_convert] = pd.DataFrame(
    [
        str_via_cat_to_int(series_in=exps[colname], categories=categories)
        for colname, categories in zip(
            cols_convert,
            [
                ["relatedness", "knowledge", "objects"],
                ["visual", "audiovisual", "auditory_visual", "auditory"],
                ["none", "manual", "covert", "overt"],
                ["SPM", "FSL"],
            ],
        )
    ]
).transpose()

# Add new columns for centered mean age and centered mean age squared
exps_sdm["age_mean_c"] = exps_sdm["age_mean"].subtract(exps_sdm["age_mean"].mean())
exps_sdm["age_mean_c_2"] = exps_sdm["age_mean_c"] ** 2

# %% [markdown]
# We are now ready to write all of these columns into a new text file called `sdm_table.txt`.

# %%
# Write the relevant columns into an SDM table
exps_sdm[
    [
        "study",
        "n1",
        "t_thr",
        "threshold",
        "age_mean_c",
        "age_mean_c_2",
        "task_type",
        "modality_pres",
        "modality_resp",
        "software",
    ]
].to_csv("../results/sdm/sdm_table.txt", sep="\t", index=False)

# %% [markdown]
# With that, we have all the files that we need to perform our SDM analyses. Because we're going to estimate multiple meta-analytic models, let's define some global parameters which we can then reuse for each model. Because SDM is computationally very expensive, it helps a lot to let it run on (almost) all available cores on your machine (or, ideally, your HPC cluster if you happen to have access to one). We're automatically storing the appropriate number of cores to use in a variable called `n_threads`. We are also specifying some default values for the number of imputations (used when computing the experiment-specific effect size maps) and the number of permutations for the empirical family-wise error (FWE) correction.

# %%
# Specify no. of threads to use, no. of mean imputations, and no of. FWE permutations
n_threads = cpu_count() - 1
n_imps = 50
n_perms = 1000

# Specify statistical thresholds
thresh_voxel_p = 0.001
thresh_cluster_k = 50

# Store working directory for SDM
cwd = "../results/sdm/"

# %% [markdown]
# With that settled, let's invoke the SDM software for the very first time. We're interacting with it via the command line (using the `subprocess.run()` function). Note that for that to work, the SDM binary file (simply called `sdm`) needs to be in the current working directory or on `$PATH`. If you are interacting with this notebook inside our Docker container (e.g., on Binder), this has already been taken care of. If you are running the code on your local system, you need to make sure to add `sdm` to `$PATH` (see, e.g., [here](https://unix.stackexchange.com/a/26059) for instructions) or to replace the `sdm` bit within each `call_` with the actual path of the SDM binary.
#
# The first step in SDM, which only needs to be done once (and not for each model separetly) is to preprocess the data. This means recreating statistical maps for all experiments based on the peak coordinates and the information about their *t* scores and statistical thresholds used in the original papers. The parameters we are using here (i.e., the gray matter mask, the anisotropy and FHWM of the Gaussian smoothing kernel, and the voxel size) are all the default values from the [SDM online manual](https://www.sdmproject.com/manual/) (which, by the way, is very useful if you need additional information on how to use the software).

# %%
# Run preprocessing (specs: template, anisotropy, FWHM, mask, voxel size)
call_pp = "sdm pp gray_matter,1.0,20,gray_matter,2"
_ = run(call_pp, shell=True, cwd=cwd)

# %% [markdown]
# After the preprocessing is, you can check the results by looking at the generated HTML report (to be found at `results/sdm/pp/pp.htm`) and the recreated *t* score maps. For now, however, let's move on directly to the next step which is estimating our meta-analytic models. Our first model (`mod1`) simply computes the meta-analytic mean effect size across all experiments without any additional covariates. It should therefore resemble our ALE analysis from Notebook #01. In `mod2`, we've decided to add (and thereby, ideally, control for) four covariates of no interest: The mean age of the sample of each experiment (which we've already mean-centered above), the modality of stimulus presentation, the modality of childrens' response, and the software package used for statistical analysis by the original authors. Note that the latter three variables were converted from categorical factors to integer values above. Finally, our third model is a meta-regression in which we test if there are clusters whose meta-analytic effect size covaries with the (mean) age of the children. In other words, we are trying to test if their are linear changes in the cortical activation patterns for semantic cognition during childhood.

# %%
# Run mean analysis without covariates
call_mod1 = "sdm mod1=mi " + str(n_imps) + ",,," + str(n_threads)
_ = run(call_mod1, shell=True, cwd=cwd)

# Run mean analysis with covariates
str_covs = "age_mean_c+modality_pres+modality_resp+software"
call_mod2 = "sdm mod2=mi " + str(n_imps) + "," + str_covs + ",," + str(n_threads)
_ = run(call_mod2, shell=True, cwd=cwd)

# Run linear model for the influence of age
str_lin = "age_mean_c,0+1+0+0+0"
call_mod3 = "sdm mod3=mi_lm " + str_lin + "," + str(n_imps) + ",," + str(n_threads)
_ = run(call_mod3, shell=True, cwd=cwd)

# %% [markdown]
# We directly get *z* score and *p* value maps for each of these models. For valid inference, however, we need to correct these for multiple comparisons. Just as ALE, SDM implements this via FWE correction based on an empirical null distribution. Note that this step is computationally very intense and will take up to 12 h per model even on machines with a large number (≥ 40) cores.

# %%
# Family-wise error (FWE) correction for all models
_ = [
    run(
        "sdm perm " + mod + "," + str(n_perms) + "," + str(n_threads),
        shell=True,
        cwd=cwd,
    )
    for mod in ["mod1", "mod2", "mod3"]
]

# %% [markdown]
# Now that we've obtained the FWE-corrected maps, we can apply the combined voxel- and cluster-level thresholding procedure taht is implemented in SDM. This will allow us to see which clusters are actually meta-analytically significant. We're doing this using a voxel-level FWE-corrected threshold combined with a cluster size threshold. This is in contrast to the cluster-level FWE that we have used in ALE. The reason for this is that the cluster-level thresholding procedure implemented in SDM uses a very different logic (based on TFCE; Smith & Nichols, 2009, NeuroImage) and routinely leads to very anti-conservative results (with basically the whole brain showing significant effect sizes). However, you can easily try this out instead by replacing `corrp_voxel` with `corrp_tfce` in the following code.

# %%
# Voxel-corrected thresholding for all models
_ = [
    run(
        "sdm threshold analysis_"
        + mod
        + "/corrp_voxel, analysis_"
        + mod
        + "/"
        + mod
        + "_z,"
        + str(thresh_voxel_p)
        + ","
        + str(thresh_cluster_k),
        shell=True,
        cwd=cwd,
    )
    for mod in ["mod1", "mod2", "mod3"]
]

# %% [markdown]
# The meta-analytic results can again be glanced at in an automatically generated HTML report. These can be found in the sub-directories that were created for each model (e.g., `results/sdm/analysis_mod1/`). With that, all the work is done for the SDM software. As a last and purely cosmetic step, we're performing another round of thresholding using the custom `dual_thresholding` function which we've created back in Notebook #02. This is necessary because the thresholding function in SDM (which we've just used) doesn't *apply* the cluster-level extent threshold to the *z* score map but only uses it for the HTML report. Because we want to plot the thresholded maps later on, we create them here using the same thresholds as above. 

# %%
# Collect the filenames of the thresholded maps created in the previous step
fnames_maps = glob("../results/sdm/analysis_mod*/mod*_z_voxelCorrected*0.nii.gz")

# Apply cluster-level threshold to these maps (SDM only does this for the HTML file)
imgs = [
    dual_thresholding(
        img_z=fname,
        voxel_thresh=0.001,
        cluster_size_mm3=200,
        two_sided=False,
        fname_out=sub("_voxelCorrected.*0.nii.gz", "_thresh.nii.gz", fname),
    )
    for fname in fnames_maps
]

# %% [markdown]
# And, of course, we also want to examine the results by plotting one of the meta-analytic maps (here for `mod1`, the model without covariates) and showing the corresponding cluster table.

# %%
# Glass brain example
p = plotting.plot_glass_brain(imgs[0], display_mode="lyrz", colorbar=True)

# Cluster table example
t = reporting.get_clusters_table(imgs[0], stat_threshold=0, min_distance=1000)
display(t)
