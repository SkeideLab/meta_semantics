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

from nimare.utils import get_template
import numpy as np
from nilearn import image, regions, plotting
from os import makedirs, path
import random
from re import sub

# %%

# Load ALE template
temp = get_template("ale_2mm", mask="gm")

# Extract MNI coordinates of all gray matter voxels
x, y, z = np.where(temp.get_fdata() == 1.0)
within_mni = image.coord_transform(x=x, y=y, z=z, affine=temp.affine)
within_mni = np.array(within_mni).transpose()

# Backup this list of coordinates
makedirs("../results/fsn/", exist_ok=True)
np.savetxt("../results/fsn/within_ale_2mm.txt", within_mni, fmt="%.0f")

# %%

# Define function to generate a new Sleuth with k null studies added
def write_null(text_file, k_null, random_seed, output_dir):

    from nimare import io
    import numpy as np
    from os import makedirs, path
    import random
    from re import sub
    from shutil import copy

    # Read the original Sleuth file into a NiMARE data set
    dset = io.convert_sleuth_to_dataset(text_file, target="ale_2mm")

    # Set random seed if requested
    if random_seed:
        random.seed(random_seed)

    # Resample numbers of subjects per experiment based on the original data
    no_subjects_dset = [n[0] for n in dset.metadata["sample_sizes"]]
    no_subjects_null = random.choices(no_subjects_dset, k=k_null)

    # Do the same for the number of foci per experiment
    no_foci_dset = dset.coordinates["study_id"].value_counts().tolist()
    no_foci_null = random.choices(no_foci_dset, k=k_null)

    # Create random foci
    idx_list = [
        random.sample(range(len(within_mni)), k=k_foci) for k_foci in no_foci_null
    ]
    foci_null = [within_mni[idx] for idx in idx_list]

    # Create the destination Sleuth file
    makedirs(output_dir, exist_ok=True)
    text_file_basename = os.path.basename(text_file)
    null_file_basename = sub(
        ".txt", "_plus_k" + str(k_null) + ".txt", string=text_file_basename
    )
    null_file = output_dir + "/" + null_file_basename
    copy(text_file, null_file)

    # Open this new file
    f = open(null_file, mode="a")

    # Add all the null studies to this file
    for i in range(k_null):
        f.write(
            "\n// nullstudy"
            + str(i + 1)
            + "\n// Subjects="
            + str(no_subjects_null[i])
            + "\n"
        )
        np.savetxt(f, foci_null[i], fmt="%.3f", delimiter="\t")

    # Close the file
    f.close()

    # Return the new Sleuth file and return it as a NiMARE data set
    dset_null = io.convert_sleuth_to_dataset(null_file, target="ale_2mm")
    return dset_null


# %%

# Params
text_file = "../results/ale/all.txt"
voxel_thresh = 0.001
cluster_thresh = 0.01
n_iters = 1000
k_null_max = 100

from nimare import io, meta, correct
from nilearn import regions
from scipy.stats import norm

# Perform ALE analysis for the original Sleuth file
dset = io.convert_sleuth_to_dataset(text_file=text_file, target="ale_2mm")
ale = meta.cbma.ALE()
res = ale.fit(dset)
corr = correct.FWECorrector("montecarlo", voxel_thresh=voxel_thresh, n_iters=n_iters)
cres = corr.transform(result=res)

# Extract thresholded cluster image
cluster_thresh_z = threshold = norm.ppf(1 - 0.01 / 2)

# Create separate maps for all significant clusters
img_orig = cres.get_map("z_level-cluster_corr-FWE_method-montecarlo")
img_orig = image.threshold_img(img_orig, threshold=cluster_thresh_z)
imgs_orig, _ = regions.connected_regions(img_orig, min_region_size=0)

# %%

# # Loop over clusters
# for i in range(imgs_clusters_orig.shape[3]):

k_null_max = 200

i = 2

img_region = image.index_img(imgs_orig, index=i)

list_k = np.array([0, k_null_max], dtype="int")
list_sig = np.array([True, False], dtype="bool")

while True:

    current_k = np.mean([list_k[list_sig].max(), list_k[~list_sig].min()], dtype="int")

    # # Compute new ALE with the maximum number of null studies added
    # dset_null = write_null(
    #     text_file=text_file,
    #     k_null=current_k,
    #     random_seed=1234,
    #     output_dir="../results/fsn/all",
    # )
    # res_null = res = ale.fit(dset_null)
    # cres_null = corr.transform(result=res_null)

    # # Check if the cluster remained significant
    # img_null = cres_null.get_map("z_level-cluster_corr-FWE_method-montecarlo")
    # img_null = image.threshold_img(img_null, threshold=cluster_thresh_z)
    # img_check = image.math_img("img1 * img2", img1=img_region, img2=img_null)
    # region_significant = np.any(img_check.get_fdata())

    region_significant = current_k < 2

    list_k = np.append(list_k, current_k)
    list_sig = np.append(list_sig, region_significant)

    # Break as soon as we're sampling the same value twice
    if len(np.unique(list_k)) != len(list_k):
        break

print(list_k)

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
