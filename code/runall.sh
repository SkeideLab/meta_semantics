#!/bin/bash
cd /home/mask_children/code
. /opt/miniconda-latest/etc/profile.d/conda.sh
conda activate nimare
JUPYTER_PARAMS=(--to=notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=-1)
jupyter nbconvert "${JUPYTER_PARAMS[@]}" nb01_ale.ipynb
jupyter nbconvert "${JUPYTER_PARAMS[@]}" nb04_sdm.ipynb
