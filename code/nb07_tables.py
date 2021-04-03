# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: 'Python 3.6.13 64-bit (''nimare'': conda)'
#     metadata:
#       interpreter:
#         hash: 4514b7299076417b8e187233bf4c34d62c86c82f88944b861e2dca408b3b9212
#     name: python3
# ---

# %%
from glob import glob
from os import makedirs, path, remove
from re import sub
from subprocess import run

import numpy as np
import pandas as pd
from atlasreader import get_statmap_info
from nilearn import image, reporting
from nimare.utils import mni2tal

# %%
# Read included experiments (Table 1)
exps = pd.read_json("../results/exps.json")
exps["foci"] = [np.array(foci, dtype="float") for foci in exps["foci"]]
exps["n_foci"] = [len(foci) for foci in exps["foci"]]

# Compute summary statistics
_ = [
    print(
        col + ":",
        "sum",
        exps[col].sum(),
        "mean",
        exps[col].mean(),
        "median",
        exps[col].median(),
        "min",
        exps[col].min(),
        "max",
        exps[col].max(),
    )
    for col in ["n", "age_mean", "age_min", "age_max", "n_foci"]
]

# Compute weighted mean age
exps["age_mean_weighted"] = [age * n for age, n in zip(exps["age_mean"], exps["n"])]
print("weighted mean age:", exps["age_mean_weighted"].sum() / exps["n"].sum())

# Compute sex and handedness ratios
print(exps["sex_female"].sum() / (exps["sex_female"].sum() + exps["sex_male"].sum()))
print(exps["hand_right"].sum() / (exps["hand_right"].sum() + exps["hand_left"].sum()))

# Count peaks with and without effect sizes
tstats = exps["tstats_corr"].explode()
print(len(tstats[tstats != "p"]))
print(len(tstats[tstats != "p"]) / len(tstats))
print(len(tstats[tstats == "p"]))

# %%
# Define function to print the clusters from multiple images as a table
def combined_cluster_table(
    img_files_z=[],
    img_files_ale=[],
    stub_keys=[],
    stub_colname="Analysis",
    atlas="aal",
    td_jar=None,
    output_file="cluster_table.tsv",
):

    # Create output director
    output_dir = path.dirname(output_file)
    makedirs(output_dir, exist_ok=True)

    # Create a list of DataFrames with peak and cluster stats for each image
    df_tuples = [
        get_statmap_info(img_file, cluster_extent=0, atlas="aal", voxel_thresh=0)
        for img_file in img_files_z
    ]
    dfs = [
        pd.DataFrame(
            {
                "Cluster #": df_tuple[0]["cluster_id"],
                "Size (mm3)": df_tuple[0]["volume_mm"],
                "Cluster labels": df_tuple[0][atlas],
                "Mean z": df_tuple[0]["cluster_mean"],
                "Peak z": df_tuple[1]["peak_value"],
                "Peak X": df_tuple[1]["peak_x"],
                "Peak Y": df_tuple[1]["peak_y"],
                "Peak Z": df_tuple[1]["peak_z"],
                "Peak label": df_tuple[1][atlas],
            }
        )
        for df_tuple in df_tuples
    ]

    # Concatenate into one big DataFrame
    df = pd.concat(dfs, keys=stub_keys)

    # Reformat numerical columns
    df["Size (mm3)"] = df["Size (mm3)"].apply(lambda x: "{:,.0f}".format(x))
    cols_int = ["Cluster #", "Peak X", "Peak Y", "Peak Z"]
    df[cols_int] = df[cols_int].applymap(int)
    cols_2f = ["Mean z", "Peak z"]
    df[cols_2f] = df[cols_2f].applymap(lambda x: "{:,.2f}".format(x))

    # Do all of this again for the ALE images if requested
    if img_files_ale:
        df_tuples_ale = [
            get_statmap_info(img_file, cluster_extent=0, atlas="aal", voxel_thresh=0)
            for img_file in img_files_ale
        ]
        dfs_ale = [
            pd.DataFrame(
                {
                    "Mean ALE": df_tuple[0]["cluster_mean"],
                    "Peak ALE": df_tuple[1]["peak_value"],
                }
            )
            for df_tuple in df_tuples_ale
        ]
        df_ale = pd.concat(dfs_ale, keys=stub_keys)
        cols_3f = ["Mean ALE", "Peak ALE"]
        df_ale[cols_3f] = df_ale[cols_3f].applymap(lambda x: "{:,.3f}".format(x))
        df.insert(4, column="Mean ALE", value=df_ale["Mean ALE"])
        df.insert(6, column="Peak ALE", value=df_ale["Peak ALE"])

    # Add the stub column
    df.index = df.index.set_names([stub_colname, ""])
    df.reset_index(level=stub_colname, inplace=True)
    mask = df[stub_colname].duplicated()
    df.loc[mask.values, [stub_colname]] = ""

    # Save to CSV
    df.to_csv(output_file, sep="\t", index=False)

    return df


# %%
# Create Table 2 (ALE results)
tab2 = combined_cluster_table(
    img_files_z=[
        "../results/ale/all_z_thresh.nii.gz",
        "../results/ale/knowledge_z_thresh.nii.gz",
        "../results/ale/lexical_z_thresh.nii.gz",
        "../results/ale/objects_z_thresh.nii.gz",
    ],
    stub_keys=[
        "All experiments",
        "Semantic knowledge",
        "Lexical semantics",
        "Visual semantics",
    ],
    stub_colname="ALE analysis",
    img_files_ale=[
        "../results/ale/all_stat_thresh.nii.gz",
        "../results/ale/knowledge_stat_thresh.nii.gz",
        "../results/ale/lexical_stat_thresh.nii.gz",
        "../results/ale/objects_stat_thresh.nii.gz",
    ],
    atlas="aal",
    output_file="../results/tables/tab2.tsv",
)
display(tab2)

# %%
# Create Table 3 (SDM results)
tab3 = combined_cluster_table(
    img_files_z=[
        "../results/sdm/analysis_mod1/mod1_z_thresh.nii.gz",
        "../results/sdm/analysis_mod2/mod2_z_thresh.nii.gz",
    ],
    stub_keys=[
        "Without covariates",
        "With covariates",
    ],
    stub_colname="SDM analysis",
    atlas="aal",
    output_file="../results/tables/tab3.tsv",
)
display(tab3)

# %%
# from nimare import utils
# from nilearn import datasets, plotting

# img = image.load_img("../results/ale/all_z_thresh.nii.gz")
# coords = np.array(test[["X", "Y", "Z"]], dtype="int").T
# coords = np.row_stack([coords, np.ones_like(coords[0])])
# vox_coords = np.linalg.solve(img.affine, coords)[0:3].astype("int")

# # i, j, k = image.coord_transform(test["X"], test["Y"], test["Z"], affine=inv_affine)
# # ijk = np.array([i, j, k], dtype="int").transpose()

# atlas = datasets.fetch_atlas_talairach("ba")
# dat = atlas.maps.get_fdata()[vox_coords[0], vox_coords[1], vox_coords[2]]

# %%
# import atlasreader

# img = image.load_img("../results/ale/all_z_thresh.nii.gz")

# tab_clusters, tab_peaks = atlasreader.get_statmap_info(
#     img, cluster_extent=0, atlas=atlasreader.atlasreader._ATLASES
# )
# display(tab_clusters)
