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

# %% [markdown] pycharm={"name": "#%% md\n"}
# # Notebook #02: Subtraction Analyses

# %% pycharm={"name": "#%%\n"}
# Import modules
from nilearn import image, plotting, reporting


# Define function for performing a single ALE subtraction analysis
def run_subtraction(text_file1, text_file2, voxel_thresh, cluster_size, n_iters, output_dir):
    print('SUBTRACTION ANALYSIS FOR "' + text_file1 + '" MINUS "' +
          text_file2 + '" WITH ' + str(n_iters) + ' PERMUTATIONS')
    # Read Sleuth files
    from nimare import io, meta
    dset1 = io.convert_sleuth_to_dataset(text_file=text_file1)
    dset2 = io.convert_sleuth_to_dataset(text_file=text_file2)
    # Perform subtraction analysis
    sub = meta.cbma.ALESubtraction(n_iters=n_iters, low_memory=False)
    sres = sub.fit(dset1, dset2)
    # Create thresholded z-map
    from nilearn.glm import threshold_stats_img
    k = cluster_size // 8
    img_z = sres.get_map('z_desc-group1MinusGroup2')
    img_z_thresh, thresh_z = threshold_stats_img(img_z, alpha=voxel_thresh, height_control='fpr',
                                                 cluster_threshold=k)
    print('(thresholding subtraction map at z > ' + str(thresh_z) + ' and k > ' + str(k) + ')')
    # Save to output directory
    from os import path, makedirs
    from nibabel import save
    makedirs(output_dir, exist_ok=True)
    name1 = path.basename(text_file1).replace('.txt', '')
    name2 = path.basename(text_file2).replace('.txt', '')
    prefix = output_dir + '/' + name1 + '_minus_' + name2
    save(img_z, filename=prefix + '_z.nii.gz')
    save(img_z_thresh, filename=prefix + '_z_tresholded.nii.gz')


# Create dictionary for which subtraction analyses to run
subtrs = dict({'../results/ale/knowledge.txt': '../results/ale/nknowledge.txt'})#,
               # '../results/ale/lexical.txt': '../results/ale/nlexical.txt',
               # '../results/ale/objects.txt': '../results/ale/nobjects.txt',
               # '../results/ale/older.txt': '../results/ale/younger.txt'})

# Use the function to perform the actual analyses
for key, value in zip(subtrs.keys(), subtrs.values()):
    run_subtraction(text_file1=key, text_file2=value, voxel_thresh=0.01, cluster_size=200,
                    n_iters=10000, output_dir='../results/subtraction')
    run_subtraction(text_file1=key, text_file2=value, voxel_thresh=0.01, cluster_size=200,
                    n_iters=10000, output_dir='../results/subtraction2')

# %% pycharm={"name": "#%%\n"}
# Glass brain example
img = image.load_img('../results/subtraction/knowledge_minus_nknowledge_z_tresholded.nii.gz')
p = plotting.plot_glass_brain(img, display_mode='lyrz')

# Cluster table example
t = reporting.get_clusters_table(img, stat_threshold=0, min_distance=1000)
t.style.format({'X': '{:.0f}', 'Y': '{:.0f}', 'Z': '{:.0f}', 'Peak Stat': '{:.2f}'}).hide_index()

# %% pycharm={"name": "#%%\n"}
# Glass brain example
img = image.load_img('../results/subtraction2/knowledge_minus_nknowledge_z_tresholded.nii.gz')
p1 = plotting.plot_glass_brain(img, display_mode='lyrz')

# Cluster table example
t = reporting.get_clusters_table(img, stat_threshold=0, min_distance=1000)
t.style.format({'X': '{:.0f}', 'Y': '{:.0f}', 'Z': '{:.0f}', 'Peak Stat': '{:.2f}'}).hide_index()

# %% pycharm={"name": "#%%\n"}
# Glass brain example
img = image.load_img('../results/subtraction3/knowledge_minus_nknowledge_z_tresholded.nii.gz')
p2 = plotting.plot_glass_brain(img, display_mode='lyrz')

# Cluster table example
t = reporting.get_clusters_table(img, stat_threshold=0, min_distance=1000)
t.style.format({'X': '{:.0f}', 'Y': '{:.0f}', 'Z': '{:.0f}', 'Peak Stat': '{:.2f}'}).hide_index()
