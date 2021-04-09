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

    # Add ALE values if available
    if img_files_ale:
        df_tuples_ale = [
            get_statmap_info(img_file, cluster_extent=0, atlas="aal", voxel_thresh=0)
            if img_file
            else (
                pd.DataFrame({"cluster_mean": [float("nan")]}),
                pd.DataFrame({"peak_value": [float("nan")]}),
            )
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
        for df, df_ale in zip(dfs, dfs_ale):
            df.insert(4, column="Mean ALE", value=df_ale["Mean ALE"])
            df.insert(6, column="Peak ALE", value=df_ale["Peak ALE"])

    # Concatenate into one big DataFrame
    df = pd.concat(dfs, keys=stub_keys)

    # Reformat numerical columns
    df["Size (mm3)"] = df["Size (mm3)"].apply(lambda x: "{:,.0f}".format(x))
    cols_int = ["Cluster #", "Peak X", "Peak Y", "Peak Z"]
    df[cols_int] = df[cols_int].applymap(int)
    cols_2f = ["Mean z", "Peak z"]
    df[cols_2f] = df[cols_2f].applymap(lambda x: "{:,.2f}".format(x))
    if img_files_ale:
        cols_3f = ["Mean ALE", "Peak ALE"]
        df[cols_3f] = df[cols_3f].applymap(lambda x: "{:,.3f}".format(x))
        df[cols_3f] = df[cols_3f].replace("nan", "")

    # Add the stub column
    df.index = df.index.set_names([stub_colname, ""])
    df.reset_index(level=stub_colname, inplace=True)
    mask = df[stub_colname].duplicated()
    df.loc[mask.values, [stub_colname]] = ""

    # Save to CSV
    df.to_csv(output_file, sep="\t", index=False)

    return df


# %%
# Create Table 2 (ALE & SDM results)
tab2 = combined_cluster_table(
    img_files_z=[
        "../results/ale/all_z_thresh.nii.gz",
        "../results/sdm/analysis_mod1/mod1_z_thresh.nii.gz",
        "../results/sdm/analysis_mod2/mod2_z_thresh.nii.gz",
    ],
    stub_keys=[
        "Activation likelihood estimation",
        "Seed-based d mapping",
        "With covariates",
    ],
    stub_colname="ALE analysis",
    img_files_ale=[
        "../results/ale/all_stat_thresh.nii.gz",
        None,
        None,
    ],
    atlas="aal",
    output_file="../results/tables/tab2.tsv",
)
display(tab2)

# %%
# Create Table 3 (task category ALEs)
tab3 = combined_cluster_table(
    img_files_z=[
        "../results/ale/knowledge_z_thresh.nii.gz",
        "../results/ale/lexical_z_thresh.nii.gz",
        "../results/ale/objects_z_thresh.nii.gz",
    ],
    stub_keys=[
        "Knowledge",
        "Relatedness",
        "Objects",
    ],
    stub_colname="ALE analysis",
    img_files_ale=[
        "../results/ale/knowledge_stat_thresh.nii.gz",
        "../results/ale/lexical_stat_thresh.nii.gz",
        "../results/ale/objects_stat_thresh.nii.gz",
    ],
    atlas="aal",
    output_file="../results/tables/tab3.tsv",
)
display(tab3)

# %%
# Create Table 4 (differences between task categories)
tab4 = combined_cluster_table(
    img_files_z=[
        "../results/subtraction/knowledge_minus_nknowledge_z_thresh.nii.gz",
        "../results/subtraction/lexical_minus_nlexical_z_thresh.nii.gz",
        "../results/subtraction/objects_minus_nobjects_z_thresh.nii.gz",
    ],
    stub_keys=[
        "Knowledge > (relatedness + objects)",
        "Relatedness > (knowledge + objects)",
        "Objects > (knowledge + relatedness)",
    ],
    stub_colname="ALE subtraction",
    atlas="aal",
    output_file="../results/tables/tab4.tsv",
)
display(tab4)
