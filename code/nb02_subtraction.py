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
#     display_name: PyCharm (mask_children)
#     language: python
#     name: pycharm-2e5bb5f0
# ---

# %% [markdown]
# # Notebook #02: Subtraction Analyses

# %%
# Define helper function for dual threshold based on voxel-p and cluster size (in mm3)
def dual_thresholding(img_z, voxel_thresh, cluster_size, two_sided=True, filename=None):
    from nilearn.glm import threshold_stats_img
    k = cluster_size // 8
    img_z_thresh, thresh_z = threshold_stats_img(img_z, alpha=voxel_thresh, height_control='fpr',
                                                 cluster_threshold=k, two_sided=two_sided)
    print('THRESHOLDED IMAGE AT Z > ' + str(thresh_z) + ' (P = ' + str(voxel_thresh) +
          ') AND K > ' + str(k) + ' (' + str(cluster_size) + ' mm3)')
    if filename:
        from nibabel import save
        save(img_z_thresh, filename=filename)
    return img_z_thresh


# Define function for performing a single ALE subtraction analysis
def run_subtraction(text_file1, text_file2, voxel_thresh, cluster_size, n_iters, output_dir):
    print('SUBTRACTION ANALYSIS FOR "' + text_file1 + '" VS. "' +
          text_file2 + '" WITH ' + str(n_iters) + ' PERMUTATIONS')
    # Read Sleuth files
    from nimare import io, meta
    dset1 = io.convert_sleuth_to_dataset(text_file=text_file1)
    dset2 = io.convert_sleuth_to_dataset(text_file=text_file2)
    # # Set a random seed to make the results reproducible
    # from numpy import random
    # random.seed(1234)
    # Perform subtraction analysis
    sub = meta.cbma.ALESubtraction(n_iters=n_iters, low_memory=False)
    sres = sub.fit(dset1, dset2)
    # Save unthresholded z-map
    from os import path, makedirs
    from nibabel import save
    img_z = sres.get_map('z_desc-group1MinusGroup2')
    makedirs(output_dir, exist_ok=True)
    name1 = path.basename(text_file1).replace('.txt', '')
    name2 = path.basename(text_file2).replace('.txt', '')
    prefix = output_dir + '/' + name1 + '_minus_' + name2
    save(img_z, filename=prefix + '_z.nii.gz')
    # Create thresholded z-map and save
    dual_thresholding(img_z=img_z, voxel_thresh=voxel_thresh, cluster_size=cluster_size,
    two_sided=True, filename=prefix + '_z_tresholded.nii.gz')


if __name__ == "__main__":
    # Create dictionary for which subtraction analyses to run
    subtrs = dict({'../results/ale/knowledge.txt': '../results/ale/nknowledge.txt'})  # ,
    # '../results/ale/lexical.txt': '../results/ale/nlexical.txt',
    # '../results/ale/objects.txt': '../results/ale/nobjects.txt',
    # '../results/ale/older.txt': '../results/ale/younger.txt'})
    # Use the function to perform the actual analyses
    for key, value in zip(subtrs.keys(), subtrs.values()):
        run_subtraction(text_file1=key, text_file2=value, voxel_thresh=0.01, cluster_size=300,
                        n_iters=1000, output_dir='../results/subtraction')
        run_subtraction(text_file1=key, text_file2=value, voxel_thresh=0.01, cluster_size=300,
                        n_iters=1000, output_dir='../results/subtraction2')

# %%
if __name__ == "__main__":

    # Import modules
    from nilearn import image, plotting, reporting

    # Glass brain example
    img = image.load_img('../results/subtraction/knowledge_minus_nknowledge_z_tresholded.nii.gz')
    p = plotting.plot_glass_brain(img, display_mode='lyrz', vmax=4, colorbar=True)

    # Cluster table example
    t = reporting.get_clusters_table(img, stat_threshold=0, min_distance=1000)
    t.style.format({'X': '{:.0f}', 'Y': '{:.0f}', 'Z': '{:.0f}',
                    'Peak Stat': '{:.2f}'}).hide_index()
    print(t)

# %%
if __name__ == "__main__":

    # Import modules
    from nilearn import image, plotting, reporting

    # Glass brain example
    img = image.load_img('../results/subtraction2/knowledge_minus_nknowledge_z_tresholded.nii.gz')
    p1 = plotting.plot_glass_brain(img, display_mode='lyrz', vmax=4, colorbar=True)

    # Cluster table example
    t = reporting.get_clusters_table(posimg, stat_threshold=0, min_distance=1000)
    t.style.format({'X': '{:.0f}', 'Y': '{:.0f}', 'Z': '{:.0f}',
                    'Peak Stat': '{:.2f}'}).hide_index()
    print(t)
