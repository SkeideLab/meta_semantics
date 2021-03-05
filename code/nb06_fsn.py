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

from nimare import io, utils
import numpy as np
import pandas as pd
from nilearn import image, plotting

# %%

# Read table of experiments from ALE analysis
exps = pd.read_json("../results/exps.json")
exps["foci"] = [np.array(foci, dtype="float") for foci in exps["foci"]]

# %%

# Load ALE template
temp = utils.get_template("ale_2mm", mask="gm")

# Extract MNI coordinates of all gray matter voxels
x, y, z = np.where(temp.get_fdata() == 1.0)
within_mni = image.coord_transform(x=x, y=y, z=z, affine=temp.affine)
within_mni = np.array(within_mni)


within_mni[:,0]

# %%

k = 10

dset = io.convert_sleuth_to_dataset("../results/ale/all.txt")
ns = dset.metadata['sample_sizes'].apply(lambda x: x[0])


