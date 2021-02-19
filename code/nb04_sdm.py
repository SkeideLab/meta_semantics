# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# Load modules
from os import path, makedirs
import pandas as pd
import numpy as np
from scipy import stats

#%%
# Unpack SDM software
if not path.isfile('../results/sdm/sdm'):
    import tarfile
    from shutil import move
    sdm_tar = tarfile.open('../software/SdmPsiGui-linux64-v6.21.tar')
    sdm_tar.extractall('../results/')
    sdm_tar.close()
    move('../results/SdmPsiGui-linux64-v6.21/', '../results/sdm/')

# %%
# Read table of experiments from ALE analysis
exps = pd.read_pickle('../results/exps.pickle')

#%%
# Extract test statistics of individal foci
exps['tstats'] = [
    # If there are t-values, we can use them directly
    foci[:, 3] if foci_stat == 'tstat'
    # If there are z-values, we convert them to t-values; if none of these, use '[p]ositive'
    else (stats.t.ppf(stats.norm.cdf(foci[:, 3]), df=n - 1) if foci_stat == 'zstat'
          else ['p'] * foci.shape[0])
    for foci, foci_stat, n
    in zip(exps['foci'], exps['foci_stat'], exps['n'])
]

# Add new test statistics back to the foci
exps['foci_sdm'] = [np.c_[foci, tstats]
                    for foci, tstats
                    in zip(exps['foci_mni'], exps['tstats'])]

# Write the foci of each experiment to a text file
_ = exps.apply(lambda x: np.savetxt(
    fname='../results/sdm/home/meta/' + x['experiment'] + '.other_mni.txt',
    X=x['foci_sdm'], fmt='%s', delimiter=','), axis=1)

# %%
# Convert voxel-wise thresholds from str to fload
exps[["thresh_vox_z",
      "thresh_vox_t",
      "thresh_vox_p"]] = exps[["thresh_vox_z",
                               "thresh_vox_t",
                               "thresh_vox_p"]].apply(lambda x: x.to_numpy(dtype='float'))

# Determine t-value thresholds of experiments
exps['t_thr'] = [
    # If there is a t-value threshold, we can use it directly
    t if not np.isnan(t)
    # If there is a z-value threshold, we convert it to a t-value
    else (stats.t.ppf(stats.norm.cdf(z), df=n - 1) if not np.isnan(z)
          # If there is a p-value threshold, we convert to a t-value
          else (abs(stats.t.ppf(p, df=n - 1)) if not np.isnan(p)
                # If none of these, use the lowest significant t-value if available
                else tstats.min()))
    for t, z, p, n, tstats in zip(exps['thresh_vox_t'],
                                  exps['thresh_vox_z'],
                                  exps['thresh_vox_p'],
                                  exps['n'],
                                  exps['tstats'])
]

# Write the relevant columns into an SDM table
makedirs('../results/sdm/home/meta', exist_ok=True)
exps_sdm = exps.rename(columns=({"experiment": "study", "n": "n1"}))
exps_sdm[["study",
          "n1",
          "t_thr",
          "threshold"]].to_csv('../results/sdm/home/meta/sdm_table.txt', sep=' ', index=False)

# %%
# Run preprocessing with SDM
from subprocess import run
run('../../sdm', cwd='../results/sdm/home/meta/')
