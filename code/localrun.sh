#!/bin/bash

# JUPYTER_PARAMS=(--to=notebook --execute --inplace \
#     --ExecutePreprocessor.kernel_name=python3 \
#     --ExecutePreprocessor.timeout=-1)
# jupyter nbconvert "${JUPYTER_PARAMS[@]}" nb01_ale.ipynb
# jupyter nbconvert "${JUPYTER_PARAMS[@]}" nb04_sdm.ipynb

python3 nb01_ale.py
python3 nb02_subtraction.py
python3 nb03_adults.py
#python3 nb04_sdm.py
#python3 nb05_jackknife.py
