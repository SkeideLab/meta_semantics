# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.2
# ---

# %%

# Define function to generate a new data set with k null studies added
def generate_null(
    text_file="foci.txt",
    space="ale_2mm",
    k_null=200,
    random_seed=None,
    output_dir="./",
):

    from nimare import io, utils
    import numpy as np
    from os import makedirs, path
    import random
    from re import sub
    from shutil import copy

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
    text_file_basename = os.path.basename(text_file)
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

# Params
text_file = "../results/ale/all.txt"
space = "ale_2mm"
voxel_thresh = 0.001
cluster_thresh = 0.01
n_iters = 1000
k_null_max = 200
random_seed = 1234
output_dir = "../results/fsn/all"

from nimare import io, meta, correct
from nilearn import regions, image
from scipy.stats import norm
import numpy as np

# Specify ALE and FWE transformers
ale = meta.cbma.ALE()
corr = correct.FWECorrector("montecarlo", voxel_thresh=voxel_thresh, n_iters=n_iters)

# Perform ALE analysis for the original Sleuth file
dset_orig = io.convert_sleuth_to_dataset(text_file=text_file, target=space)
res_orig = ale.fit(dset_orig)
cres_orig = corr.transform(result=res_orig)

# Store a list of the study IDs of the original studies
study_ids_orig = dset_orig.metadata["id"].tolist()

# Extract thresholded cluster image
cluster_thresh_z = norm.ppf(1 - cluster_thresh / 2)

# Create separate maps for all significant clusters
img_orig = cres_orig.get_map("z_level-cluster_corr-FWE_method-montecarlo")
img_orig = image.threshold_img(img_orig, threshold=cluster_thresh_z)
imgs_orig, _ = regions.connected_regions(img_orig, min_region_size=0)

# %%

# Create a new data set with null studies added
dset_null = generate_null(
    text_file=text_file,
    space=space,
    k_null=k_null_max,
    random_seed=random_seed,
    output_dir=output_dir,
)

# %%

# # Loop over clusters
# for cl in range(imgs_clusters_orig.shape[3]):

# Current cluster
cl = 0

# Extract the map with only the current cluster
img_cluster = image.index_img(imgs_orig, index=cl)

# Create lists to keep track of the sampleing process:
# 1) How many null studies did we use at this step?
list_k = np.array([0, k_null_max], dtype="int")
# 2) Was our cluster at this step significant or not?
list_sig = np.array([True, False], dtype="bool")

# Start sampling
while True:

    # Compute how many null studies to use for the current iteration
    current_k = np.mean([list_k[list_sig].max(), list_k[~list_sig].min()], dtype="int")

    # Create a list of these null studies
    study_ids_null = ["nullstudy" + str(x + 1) + "-" for x in range(current_k)]

    # Create a new data set where these are added to the original studies
    study_ids = study_ids_orig + study_ids_null
    dset_current = dset_null.slice(study_ids)

    # Compute the ALE for the current data set
    res_current = res = ale.fit(dset_current)
    cres_current = corr.transform(result=res_current)

    # Check if the cluster remained significant
    img_current = cres_current.get_map("z_level-cluster_corr-FWE_method-montecarlo")
    img_current = image.threshold_img(img_current, threshold=cluster_thresh_z)
    img_check = image.math_img("img1 * img2", img1=img_cluster, img2=img_current)
    cluster_significant = np.any(img_check.get_fdata())

    # Add the new information to the lists
    list_k = np.append(list_k, current_k)
    list_sig = np.append(list_sig, cluster_significant)

    # Stop as soon as k doesn't change animore. We've found our FSN!
    if len(np.unique(list_k)) != len(list_k):
        break

# Take a look at the sampling process
print(list_k)
print(list_sig)

# %%

# from nilearn import regions

# img = image.load_img(
#     "../results/ale/all_logp_level-cluster_corr-FWE_method-montecarlo.nii.gz"
# )
# img_thresh = image.threshold_img(img, threshold=norm.ppf(1 - 0.01 / 2))
# plotting.plot_glass_brain(img_thresh, colorbar=True)

# img_reg, _ = regions.connected_regions(img_thresh)
# plotting.plot_glass_brain(image.index_img(img_reg, index=0))

# # %%

# # Write data set with added null studies
# dset_null = write_null(
#     text_file="../results/ale/lexical.txt",
#     k_null=10,
#     random_seed=1234,
#     output_dir="../results/fsn/lexical/",
# )

# voxel_thresh = 0.01
# cluster_thresh = 0.05
# n_iters = 10
# k_null = 10

# from nimare import meta, correct
# from scipy.stats import norm
# from nibabel import save

# studies_all = dset_null.metadata["study_id"].tolist()

# list_studies_remove = [
#     ["nullstudy" + str(i + 1) for i in range(k_start, k_null)]
#     for k_start in reversed(range(k_null))
# ]

# list_studies_keep = [
#     list(set(studies_all) - set(studies_remove))
#     for studies_remove in list_studies_remove
# ]

# dsets_null = [dset_null.slice(studies_keep) for studies_keep in list_studies_keep]


# [len(x.metadata) for x in dsets_null]


# print(studies)

# # Compute all ALE analysis for up to k null studies
# ale = meta.cbma.ALE()
# res = ale.fit(dset_null)

# # Perform the FWE correction
# corr = correct.FWECorrector("montecarlo", voxel_thresh=voxel_thresh, n_iters=n_iters)
# cres = corr.transform(result=res)

# img_clust = cres.get_map("z_level-cluster_corr-FWE_method-montecarlo")
# cluster_thresh_z = norm.ppf(1 - cluster_thresh / 2)
# formula = "np.where(img > " + str(cluster_thresh_z) + ", 1, 0)"
# img_mask = image.math_img(formula, img=img_clust)
# # save(img_mask, filename="")

# plotting.plot_glass_brain(img_clust, colorbar=True)
# plotting.plot_glass_brain(img_mask, colorbar=True)

# %%

# Define one more function to write the null studies *and* run the ALE
def run_ale_null(
    text_file, k_null, random_seed, voxel_thresh, cluster_thresh, n_iters, output_dir
):

    from nb01_ale import run_ale

    # Write the Sleuth file
    null_file = write_null(
        text_file=text_file,
        k_null=k_null,
        random_seed=random_seed,
        output_dir=output_dir,
    )

    # Run the ALE
    run_ale(
        text_file=null_file,
        voxel_thresh=voxel_thresh,
        cluster_thresh=cluster_thresh,
        n_iters=n_iters,
        output_dir=output_dir,
    )


# %%

# Specify Sleuth files for which we want run FSN analyses
text_files = ["../results/ale/lexical.txt", "../results/ale/objects.txt"]

# Create output folder names based on these files
output_dirs = [
    sub("ale/", "fsn/", path.splitext(text_file)[0]) for text_file in text_files
]

# Create some reproducible random seeds
random.seed(1234)
random_seeds = [random.randint(0, 1000) for _ in range(len(text_files))]

# %%


_ = [
    run_ale_null(
        text_file=text_file,
        k_null=1,
        random_seed=random_seed,
        voxel_thresh=0.001,
        cluster_thresh=0.01,
        n_iters=5,
        output_dir=output_dir + "/",
    )
    for text_file, random_seed, output_dir in zip(text_files, random_seeds, output_dirs)
]
