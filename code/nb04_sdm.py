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
from subprocess import run
from multiprocessing import cpu_count

# %%
# Unpack SDM software
if not path.isfile('../results/sdm/sdm'):
    from glob import glob
    import tarfile
    from shutil import move
    fname_tar = glob('../software/SdmPsiGui-*')[0]
    tar_sdm = tarfile.open(fname_tar)
    tar_sdm.extractall('../results/')
    tar_sdm.close()
    move('../results/SdmPsiGui-linux64-v6.21/', '../results/sdm/')

# %%
# Read table of experiments from ALE analysis
exps = pd.read_pickle('../results/exps.pickle')

# %%
# Extract test statistics of individal foci
exps['tstats'] = [
    # If there are t-values, we can use them directly
    foci[:, 3] if foci_stat == 'tstat'
    # If there are z-values, we convert them to t-values
    else (stats.t.ppf(stats.norm.cdf(foci[:, 3]), df=n - 1) if foci_stat == 'zstat'
          # If neither of these, write NaNs
          else np.full(foci.shape[0], np.nan))
    for foci, foci_stat, n
    in zip(exps['foci'], exps['foci_stat'], exps['n'])
]

# Replace infinite t-values with the maximum t-value of the experiment
exps['tstats'] = [np.where(np.isinf(tstats), np.max(tstats[tstats != np.inf]), tstats)
                  for tstats in exps['tstats']]

# Replace missing t-values with [p]ositive
exps['tstats'] = [np.where(np.isnan(tstats), 'p', tstats) for tstats in exps['tstats']]

# Add new test statistics back to the foci
exps['foci_sdm'] = [np.c_[foci, tstats]
                    for foci, tstats
                    in zip(exps['foci_mni'], exps['tstats'])]

# Write the foci of each experiment to a text file
makedirs('../results/sdm/home/meta', exist_ok=True)
_ = exps.apply(lambda x: np.savetxt(
    fname='../results/sdm/home/meta/' + x['experiment'] + '.other_mni.txt',
    X=x['foci_sdm'], fmt='%s', delimiter=','), axis=1)

# %%
# Convert some columns from str to fload
cols_thresh = ['thresh_vox_z', 'thresh_vox_t', 'thresh_vox_p']
exps[cols_thresh] = exps[cols_thresh].apply(pd.to_numeric, errors='coerce')

# Determine t-value thresholds of experiments
exps['t_thr'] = [
    # If there is a t-value threshold, we can use it directly
    t if not np.isnan(t)
    # If there is a z-value threshold, we convert it to a t-value
    else (stats.t.ppf(stats.norm.cdf(z), df=n - 1) if not np.isnan(z)
          # If there is a p-value threshold, we convert to a t-value
          else (abs(stats.t.ppf(p, df=n - 1)) if not np.isnan(p)
                # If none of these, use the lowest significant t-value if available
                else pd.to_numeric(tstats).min()))
    for t, z, p, n, tstats in zip(exps['thresh_vox_t'],
                                  exps['thresh_vox_z'],
                                  exps['thresh_vox_p'],
                                  exps['n'],
                                  exps['tstats'])
]

# %%

# Copy the table and rename some columns
exps_sdm = exps.rename(columns=({'experiment': 'study', 'n': 'n1'}))


# Define function to convert string variables to integers (as dummies for categories)
def str_via_cat_to_int(series_in, categories):
    # Convert strings to category codes (in the provided order), starting at 1
    from pandas import Categorical
    series_out = Categorical(series_in).set_categories(new_categories=categories).codes + 1
    # Add another category code for any leftover categories
    series_out[series_out == 0] = series_out.max() + 1
    return series_out


# Apply this function to convert some columns to integers
cols_convert = ['task_type', 'modality_pres', 'modality_resp', 'software']
exps_sdm[cols_convert] = pd.DataFrame([
    str_via_cat_to_int(series_in=exps[colname], categories=categories)
    for colname, categories in zip(cols_convert,
                                   [['lexical', 'knowledge', 'objects'],
                                    ['visual', 'audiovisual', 'auditory_visual', 'auditory'],
                                    ['none', 'manual', 'covert', 'overt'],
                                    ['SPM', 'FSL']])]).transpose()

# Add new columns for centered mean age and centered mean age squared
exps_sdm['age_mean_c'] = exps_sdm['age_mean'].subtract(exps_sdm['age_mean'].mean())
exps_sdm['age_mean_c_2'] = exps_sdm['age_mean_c'] ** 2

# Write the relevant columns into an SDM table
exps_sdm[['study', 'n1',
          't_thr',
          'threshold',
          'age_mean_c',
          'age_mean_c_2',
          'task_type',
          'modality_pres',
          'modality_resp',
          'software']].to_csv('../results/sdm/home/meta/sdm_table.txt', sep='\t', index=False)

# %%
# Store path of the SDM binary and the no. of threads, mean imputations, and cFWE permutations
fname_sdm = '../../sdm'
n_threads = cpu_count() - 1
n_imps = 10
n_perms = 10

# Run preprocessing (specs: template, anisotropy, FWHM, mask, voxel size)
run(fname_sdm + ' pp gray_matter,1.0,20,gray_matter,2',
    shell=True, cwd='../results/sdm/home/meta/')

# %%

# Run mean analysis without covariates (specs: imputations, threads)
run(fname_sdm + ' mean=mi ' + str(n_imps) + ',,,' + str(n_threads),
    shell=True, cwd='../results/sdm/home/meta/')

# Run mean analysis with covariates (specs: imputations, covariates, threads)
run(fname_sdm + ' covs=mi 50, age_mean+task_type+modality_resp+software, , ,' + str(n_threads),
    shell=True, cwd='../results/sdm/home/meta/')

# Run linear model for the influence of age
run(fname_sdm + ' age=mi_lm age_mean_c+age_mean_c_2,0+1+1+0+0,50,,' + str(n_threads),
    shell=True, cwd='../results/sdm/home/meta/')

# %%

# Family-wise error (FWE) correction for all models
_ = [run(fname_sdm + ' perm ' + mod + ',' + str(n_perms) + ',' + str(n_threads),
         shell=True, cwd='../results/sdm/home/meta/')
     for mod in ['mean', 'covs', 'age']]

# %%

# Thresholding for all models
thresh_voxel_p = 0.001
thresh_cluster_k = 50
_ = [run(fname_sdm + ' threshold analysis_' + mod + '/corrp_voxel, analysis_' + mod +
         '/' + mod + '_z, ' + str(thresh_voxel_p) + ', ' + str(thresh_cluster_k),
         shell=True, cwd='../results/sdm/home/meta/')
     for mod in ['mean', 'covs', 'age']]