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
import random

# %%
# Define function to generate a new data set with k null studies added
def generate_null(
    text_file="foci.txt",
    space="ale_2mm",
    k_null=100,
    random_seed=None,
    output_dir="./",
):

    from nimare import io, utils
    import numpy as np
    from nilearn import image
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
    random_seed=None,
    output_dir="./",
):

    from nimare import io, meta, correct
    from nilearn import image
    from scipy.stats import norm
    import numpy as np
    from nibabel import nifti1, save

    # Print what we're going to do
    print(
        "\nCOMPUTING FSN FOR ALL VOXELS IN "
        + text_file
        + " (random seed: "
        + str(random_seed)
        + ")\n"
    )

    # Set random seeds if requested
    if random_seed:
        np.random.seed(random_seed)

    # Perform the original ALE
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
    k_max = len(ids_orig) * 20
    dset_null = generate_null(
        text_file=text_file,
        space=space,
        k_null=k_max,
        random_seed=random_seed,
        output_dir=output_dir,
    )

    # Extract thresholded cluster mask from the original ALE
    img_orig = cres_orig.get_map("z_level-cluster_corr-FWE_method-montecarlo")
    cluster_thresh_z = norm.ppf(1 - cluster_thresh / 2)
    img_orig = image.threshold_img(img_orig, threshold=cluster_thresh_z)
    save(img_orig, filename=output_dir + "/k0_z_thresh.nii.gz")
    img_orig = image.math_img("np.where(img > 0, 1, 0)", img=img_orig)

    # Create a list to store the images FSN at intermediate steps
    imgs_k = [img_orig]

    # Initialize the number of null studies
    k = 0

    # Iteratively add null studies
    while True:

        # Add a null study
        k = k + 1

        # Print message
        print("Simulating ALE for k = " + str(k) + " null studies added.")

        # Create a new data set
        ids_null = ["nullstudy" + str(x + 1) + "-" for x in range(0, k)]
        ids = ids_orig + ids_null
        dset_k = dset_null.slice(ids)

        # Fit the ALE
        res_k = res = ale.fit(dset_k)
        cres_k = corr.transform(result=res_k)

        # Extract and save the thresholded cluster map
        img_k = cres_k.get_map("z_level-cluster_corr-FWE_method-montecarlo")
        img_k = image.threshold_img(img_k, threshold=cluster_thresh_z)
        save(img_k, filename=output_dir + "/k" + str(k) + "_z_thresh.nii.gz")

        # Check which voxels remained significant
        img_k = image.math_img("np.where(img > 0, 1, 0)", img=img_k)
        img_mult = image.math_img("img1 * img2", img1=img_orig, img2=img_k)

        # If there are any voxels left, store the image in our list
        if np.any(img_mult.get_fdata()):
            imgs_k.append(img_mult)
        else:
            print("No more significant voxels - terminating.\n")
            break

    # Extract data from all images as an array
    imgs_k = image.concat_imgs(imgs_k)
    dat_k = imgs_k.get_fdata()

    # Create empty array for the per-voxel FSN
    dat_fsn = np.zeros(dat_k.shape[0:3])

    # Extract per-voxel FSN (i.e., the first image where it wasn't significant)
    for x in range(dat_k.shape[0]):
        for y in range(dat_k.shape[1]):
            for z in range(dat_k.shape[2]):
                voxel_significant = dat_k[x, y, z, :]
                dat_fsn[x, y, z] = np.argmax(voxel_significant == 0)

    # Create a Nifti map with the per-voxel FSN
    img_fsn = nifti1.Nifti1Image(dat_fsn, affine=img_orig.affine)
    save(img_fsn, filename=output_dir + "/fsn.nii.gz")
    return img_fsn


# %%
# List Sleuth files for which we want to perform an FSN analysis
prefixes = ["all", "knowledge", "lexical", "objects"]
prefixes = ["knowledge", "objects"]
text_files = ["../results/ale/" + prefix + ".txt" for prefix in prefixes]

# Create output directory based on these filenames
output_dirs = ["../results/fsn_full/" + prefix + "/" for prefix in prefixes]

# How many different filedrawers to compute for each text file?
nr_filedrawers = 10
filedrawers = ["filedrawer" + str(fd) for fd in range(nr_filedrawers)]

# Create a reproducible random seed for each filedrawer
random.seed(1234)
random_seeds = random.sample(range(0, 1000), k=nr_filedrawers)

# Use our function to compute multiple filedrawers for each text file
imgs_fsn = [
    [
        compute_fsn(
            text_file=text_file,
            space="ale_2mm",
            voxel_thresh=0.001,
            cluster_thresh=0.01,
            n_iters=1000,
            random_seed=random_seed,
            output_dir=output_dir + filedrawer,
        )
        for random_seed, filedrawer in zip(random_seeds, filedrawers)
    ]
    for text_file, output_dir in zip(text_files, output_dirs)
]
