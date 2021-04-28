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
# # Notebook #03: Comparison of Semantic Cognition in Children and Adults
#
# *Created April 2021 by Alexander Enge* ([enge@cbs.mpg.de](mailto:enge@cbs.mpg.de))
#
# Here we compare our ALE results for semantic cognition in children to the results from a previous meta-analysis that also used ALE to investigate semantic cognition in adults (Jackson, 2021, *NeuroImage*). The original author of this analysis was so kind to provide us with their Sleuth file, which contains the coordinates from more than 400 fMRI experiments with adults. Using the same techniques functions as in Notebook #01, we recreate the ALE analysis for the adult data and then, using the same functions as in Notebook #02, we contrast the two groups against one another.
#
# Let's again start by loading all the packages we need. Note that we're also importing two of the custom functions which we have created in the first two notebooks.

# %%
from os import makedirs
from shutil import copy

from IPython.display import display
from nibabel import save
from nilearn import image, plotting, reporting
from nimare import io

from nb01_ale import run_ale
from nb02_subtraction import run_subtraction

# %% [markdown]
# We create a new output directory and out our two pre-existing Sleuth files there: The child-specific Sleuth file was created with the help of Notebook #01 and the adult-specific Sleuth file was kindly provided to us by Dr Rebecca L. Jackson from MRC CBU at Cambridge (UK).

# %%
# Copy Sleuth text files to the results directory
output_dir = "../results/adults/"
_ = makedirs(output_dir, exist_ok=True)
_ = copy("../results/ale/all.txt", output_dir + "children.txt")
_ = copy("../data/adults/adults.txt", output_dir + "adults.txt")

# %% [markdown]
# We can use our custom ALE function to recreate the adult-specific analysis. We use the same voxel- and cluster-level thresholds as for the children to allow for a meaningful group comparison.

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

# %% [markdown]
# The group comparison can now be computed with the help of our custom function which we defined in Notebook #02.

# %%
# Perform subtraction analysis for children vs. adults
run_subtraction(
    text_file1="../results/adults/children.txt",
    text_file2="../results/adults/adults.txt",
    voxel_thresh=0.001,
    cluster_size_mm3=200,
    random_seed=1234,
    n_iters=10000,
    output_dir=output_dir,
)

# %% [markdown]
# As a cosmetic measure, we split up the resulting difference map into two separte ones: One showing only the significant clusters for children > adults and one showing only the significant clusters for adults > children. This will make it easier later on to present these two sets of clusters in separate cluster tables.

# %%
# Compute seperate difference maps for children > adults and adults > children
img_sub = image.load_img(output_dir + "children_minus_adults_z_thresh.nii.gz")
img_children_gt_adults = image.math_img("np.where(img > 0, img, 0)", img=img_sub)
img_adults_gt_children = image.math_img("np.where(img < 0, img * -1, 0)", img=img_sub)
_ = save(img_children_gt_adults, output_dir + "children_greater_adults_z_thresh.nii.gz")
_ = save(img_adults_gt_children, output_dir + "adults_greater_children_z_thresh.nii.gz")

# %% [markdown]
# Finally, we also compute a conjunction map. This map shows all the brain regions that are engaged in semantic cognition in *both* groups (but not those specific to either one of them). For these voxels, we just take the smaller of the two *z* values from both group-specific *z* score maps (Nichols et al., 2005, *NeuroImage*). We then do the same for the ALE maps so we have our conjunction maps with both *z* scores and ALE values.

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

# %% [markdown]
# Now let's look at the different maps that we've created in the previous steps. We started with adult-specific ALE analysis.

# %%
# Glass brain for adults only
p = plotting.plot_glass_brain(img_adults_z, display_mode="lyrz", colorbar=True)

# Cluster table for adults only
t = reporting.get_clusters_table(img_adults_z, stat_threshold=0, min_distance=1000)
display(t)

# %% [markdown]
# Second, let's plot the subtraction map which shows us the group differences between children and adults.

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

# Cluster table for children vs. adults
t_sub = reporting.get_clusters_table(
    img_sub, stat_threshold=0, min_distance=1000, two_sided=True
)
display(t_sub)

# %% [markdown]
# And, finally, let's also plot the conjunction map to show which clusters were engaged in semantic cognition in both children *and* adults.

# %%
# Glass brain for conjunction
p_conj = plotting.plot_glass_brain(
    img_conj_z,
    display_mode="lyrz",
    colorbar=True,
    vmax=8,
)

# Cluster table for conjunction
t_conj = reporting.get_clusters_table(img_conj_z, stat_threshold=0, min_distance=1000)
display(t_conj)
