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

# Params
text_file = "../results/ale/objects.txt"
space = "ale_2mm"
voxel_thresh = 0.001
cluster_thresh = 0.01
n_iters = 1000
k_null_max = 200
random_seed = 1234
output_dir = "../results/fsn/objects"

from nimare import io, meta, correct
from nilearn import regions, image
from scipy.stats import norm
import numpy as np

# Specify ALE and FWE transformers
ale = meta.cbma.ALE()
corr = correct.FWECorrector("montecarlo", voxel_thresh=voxel_thresh, n_iters=n_iters)

# Perform ALE analysis for the original Sleuth file
print("RECREATING ORIGINAL ALE ANALYSIS (K = 0 NULL STUDIES)")
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

# Create an empty cache to store images we've already computed
cache_imgs = dict()

# %%

# Create a dict of clusters
fsn_per_cluster = dict()

# Perform FSN sampling algorithm for each cluster
for cl in range(imgs_orig.shape[3]):

    # Extract the map with only the current cluster
    print("\nPERFORMING FSN SAMPLING FOR CLUSTER #" + str(cl) + "\n")
    img_cluster = image.index_img(imgs_orig, index=cl)

    # Create lists to keep track of the sampleing process:
    # 1) How many null studies did we use at this step?
    ks = np.array([0, k_null_max], dtype="int")
    # 2) Was our cluster at this step significant or not?
    sigs = np.array([True, False], dtype="bool")

    # We keep sampling values for k until we hit the same value twice (= our final FSN)
    while len(np.unique(ks)) == len(ks):

        # Compute how many null studies to use for the current iteration
        k = np.mean([ks[sigs].max(), ks[~sigs].min()], dtype="int")

        # See if the necessary image is alread in the cache
        key_k = "k" + str(k)

        if key_k in cache_imgs:

            # If so, just load the image from the cache
            print("Loading image from cache for k = " + str(k))
            img_k = cache_imgs[key_k]

        else:

            # Create a list of these null studies
            print("Computing ALE for k = " + str(k))
            study_ids_null = ["nullstudy" + str(x + 1) + "-" for x in range(k)]

            # Create a new data set where these are added to the original studies
            study_ids = study_ids_orig + study_ids_null
            dset_k = dset_null.slice(study_ids)

            # Compute the ALE for the current data set
            res_k = res = ale.fit(dset_k)
            cres_k = corr.transform(result=res_k)

            # Write the image to the cache so we don't have to re-compute it later
            img_k = cres_k.get_map("z_level-cluster_corr-FWE_method-montecarlo")
            cache_imgs[key_k] = img_k

        # Check if the cluster has remained significant
        img_k = image.threshold_img(img_k, threshold=cluster_thresh_z)
        img_check = image.math_img("img1 * img2", img1=img_cluster, img2=img_k)
        cluster_significant = np.any(img_check.get_fdata())

        # Add the new information to the lists
        ks = np.append(ks, k)
        sigs = np.append(sigs, cluster_significant)

    # Backup the sampling process for this cluster
    print("\nDONE SAMPLING FOR CLUSTER #" + str(cl) + ": FSN = " + str(k) + "\n")
    key_cl = "cl" + str(cl)
    fsn_per_cluster[key_cl] = ks

# We're done!
print("DONE")
print(fsn_per_cluster)


# Save to pickle
import pickle

f = open("../results/fsn/objects/fsn_results.pickle", "wb")
pickle.dump(fsn_per_cluster, f)
f.close()

f = open("../results/fsn/objects/fsn_cache.pickle", "wb")
pickle.dump(cache_imgs, f)
f.close()
