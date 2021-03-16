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
from nilearn import image, reporting
from nimare.utils import mni2tal

# %%
# Define function to print the clusters from multiple images as a table
def combined_cluster_table(
    img_files_z=[],
    img_files_ale=[],
    stub_keys=[],
    stub_colname="Analysis",
    td_jar=None,
    output_file="cluster_table.tsv",
):

    # Create output director
    output_dir = path.dirname(output_file)
    makedirs(output_dir, exist_ok=True)

    # Create a list of tables from the image files and concatenate
    imgs_z = [image.load_img(filename) for filename in img_files_z]
    tabs_z = [
        reporting.get_clusters_table(img, stat_threshold=0, min_distance=np.inf)
        for img in imgs_z
    ]
    df = pd.concat(tabs_z, keys=stub_keys)

    # Get gray matter labels from the Talairach Deamon (Lancaster et al., 1997, 2000)
    if td_jar:

        # Write a file with coordinates in Talairach space
        coords_file = output_dir + "/coords.txt"
        coords_mni = np.array(df[["X", "Y", "Z"]])
        coords_tal = mni2tal(coords_mni)
        np.savetxt(coords_file, coords_tal, fmt="%.2f", delimiter="\t")

        # Run Talairach Deamon from the command line
        call = "java -cp " + td_jar + " org.talairach.ExcelToTD 4, " + coords_file
        run(call, shell=True)

        # Read into new df
        names = ["X", "Y", "Z", "Hem", "Lobe", "Struct", "Matter", "BA"]
        df_td = pd.read_csv(coords_file + ".td", sep="\t", header=None, names=names)

        # Reformat and add to main df
        df_td["Hem"] = df_td["Hem"].str[0]
        df_td["BA"] = "(" + df_td["BA"] + ")"
        df_td["BA"] = df_td["BA"].str.replace("Brodmann area", "BA").replace("(*)", "")
        labels = df_td["Hem"] + " " + df_td["Struct"] + " " + df_td["BA"]
        labels.index = df.index
        df.insert(6, column="Nearest gray matter", value=labels)

        # Remove intermediate files
        _ = [remove(f) for f in [coords_file, coords_file + ".td"]]

    # Do the same for the additional images and add to table (if requested)
    if img_files_ale:
        imgs_ale = [image.load_img(filename) for filename in img_files_ale]
        tabs_ale = [
            reporting.get_clusters_table(img, stat_threshold=0, min_distance=np.inf)
            for img in imgs_ale
        ]
        df_ale = pd.concat(tabs_ale, keys=stub_keys)
        df_ale["Peak Stat"] = df_ale["Peak Stat"].round(3)
        df.insert(4, column="Peak ALE", value=df_ale["Peak Stat"])

    # Add the stub column
    df.index = df.index.set_names([stub_colname, ""])
    df.reset_index(level=stub_colname, inplace=True)
    mask = df[stub_colname].duplicated()
    df.loc[mask.values, [stub_colname]] = ""

    # Rename columns
    df.rename(
        columns={
            "Cluster ID": "Cluster",
            "Peak Stat": "Peak z",
        },
        inplace=True,
    )

    # Move cluster size to another position
    size = df.pop("Cluster Size (mm3)")
    df.insert(2, column="Size (mm3)", value=size)

    # Re-format numerical columns
    df[["X", "Y", "Z"]] = df[["X", "Y", "Z"]].applymap(int)
    df["Peak z"] = df["Peak z"].round(2)
    df["Size (mm3)"] = ["{:,}".format(x) for x in df["Size (mm3)"]]

    # Save the cluster table
    df.to_csv(output_file, sep="\t", index=False)

    return df


test = combined_cluster_table(
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
    td_jar="../software/talairach.jar",
    output_file="../results/tables/ale.tsv",
)

display(test)

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
