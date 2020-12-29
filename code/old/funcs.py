# Define function to run a single ALE analysis
def run_ale(fname, gingerale, p_voxel, p_cluster, perm):
    print('Performing ALE for "' + fname + '" with ' + str(perm) + ' permutations\n')
    # Run ALE from the command line
    from subprocess import Popen
    cmd_ale = 'java -cp ' + gingerale + ' org.brainmap.meta.getALE2 ' + fname + \
              ' -mask=MNI152_wb.nii -p=' + str(p_voxel) + ' -perm=' + str(perm) + \
              ' -clust=' + str(p_cluster) + ' -nonAdd'
    Popen(cmd_ale, shell=True).wait()
    # Retrieve cluster stats
    from os.path import splitext
    prefix =  splitext(fname)[0]
    perm_str = str(perm // 1000) + 'k' if perm >= 1000 else str(perm)
    cmd_cluster = 'java -cp ' + gingerale + ' org.brainmap.meta.getClustersStats ' + fname + \
                  ' ' + prefix + '_ALE.nii ' + prefix + '_p001_C01_' + perm_str + \
                  '_clust.nii -mni -p=' + prefix + '_PVal.nii -z=' + prefix + '_Z.nii'
    Popen(cmd_cluster, shell=True).wait()
