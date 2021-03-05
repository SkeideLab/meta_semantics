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
from nilearn import image
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

    # Resample numbers of subjects per experiment based on the original data
    no_subjects_dset = [n[0] for n in dset.metadata["sample_sizes"]]
    if random_seed:
        random.seed(random_seed)
    no_subjects_null = random.choices(no_subjects_dset, k=k_null)

    # Do the same for the number of foci per experiment
    no_foci_dset = dset.coordinates["study_id"].value_counts().tolist()
    if random_seed:
        random.seed(random_seed)
    no_foci_null = random.choices(no_foci_dset, k=k_null)

    # Create random foci
    if random_seed:
        random.seed(random_seed)
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

    # Return the filename of the new Sleuth file
    return null_file


#%%

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
