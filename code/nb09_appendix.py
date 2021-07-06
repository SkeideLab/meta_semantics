# -*- coding: utf-8 -*-
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
# ![SkeideLab and MPI CBS logos](../misc/header_logos.png)
#
# # Notebook #09: Appendix
#
# *Created July 2021 by Alexander Enge* ([enge@cbs.mpg.de](mailto:enge@cbs.mpg.de))
#
# Blablabla
#
# We start by loading the relevant packages.

# %%
import pandas as pd

from nb01_ale import run_ale, write_peaks_to_sleuth
from nb02_subtraction import run_subtraction

# %% [markdown]
# Blablabla

# %%
# Read table of experiments from ALE analysis
exps = pd.read_json("../results/exps.json")

# %% Define ALE analyses based on presentation and response modalities
ales = dict(
    {
        "../results/ale/visual.txt": 'modality_pres == "visual"',
        "../results/ale/nvisual.txt": 'modality_pres != "visual"',
        "../results/ale/manual.txt": 'modality_resp == "manual"',
        "../results/ale/nmanual.txt": 'modality_resp == ["covert", "overt"]',
    }
)

# Write the relevant experiments to Sleuth text files
for key, value in zip(ales.keys(), ales.values()):
    write_peaks_to_sleuth(text_file=key, df=exps, query=value)

# %%
# Compute the ALE for each Sleuth file
for key in ales.keys():
    run_ale(
        text_file=key,
        voxel_thresh=0.001,
        cluster_thresh=0.01,
        random_seed=1234,
        n_iters=1000,
        output_dir="../results/ale/",
    )

# %%
# Define subraction analyses
subtrs = dict(
    {
        "../results/ale/visual.txt": "../results/ale/nvisual.txt",
        "../results/ale/manual.txt": "../results/ale/nmanual.txt",
    }
)

# Run subtraction analyses
for key, value in zip(subtrs.keys(), subtrs.values()):
    run_subtraction(
        text_file1=key,
        text_file2=value,
        voxel_thresh=0.001,
        cluster_size_mm3=200,
        random_seed=1234,
        n_iters=20000,
        output_dir="../results/subtraction",
    )
