# Read Sleuth files
from nimare import io, meta
dset1 = io.convert_sleuth_to_dataset(text_file='older.txt')
dset2 = io.convert_sleuth_to_dataset(text_file='younger.txt')

# Perform subtraction analysis
sub = meta.cbma.ALESubtraction(n_iters=50000, low_memory=False)
sres = sub.fit(dset1, dset2)

# Save + plot
from nibabel import save
from nilearn import plotting
img_z = sres.get_map('z_desc-group1MinusGroup2')
save(img_z, filename='older_minus_younger.nii.gz')
p = plotting.plot_glass_brain('older_minus_younger.nii.gz', colorbar=True)
p.savefig('older_minus_younger.png')

# Read Sleuth files
from nimare import io, meta
dset1 = io.convert_sleuth_to_dataset(text_file='older.txt')
dset2 = io.convert_sleuth_to_dataset(text_file='younger.txt')

# Perform subtraction analysis
ale1 = meta.cbma.ALE()
res1 = ale1.fit(dset1)
ale2 = meta.cbma.ALE()
res2 = ale2.fit(dset2)

sub = meta.cbma.ALESubtraction(n_iters=10000)
sres = sub.fit(ale1, ale2)

# Save + plot
from nilearn import plotting
from nibabel import save
img_z = sres.get_map('z_desc-group1MinusGroup2')
save(img_z, filename='older_minus_younger_003.nii.gz')
p = plotting.plot_glass_brain('older_minus_younger_003.nii.gz', vmax=8, colorbar=True)
p.savefig('older_minus_younger_003.png')

