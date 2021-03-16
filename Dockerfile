FROM neurostuff/nimare:latest

# Specify user and project names
ENV NB_USER=neuro \
    PROJ_DIR=/home/neuro/mask_children \
    # Activate Conda environment
    PATH=/opt/miniconda-latest/envs/nimare/bin:$PATH \
    CONDA_DEFAULT_ENV=nimare \
    CONDA_PREFIX=/opt/miniconda-latest/envs/nimare

# Set working directory
RUN mkdir -p ${PROJ_DIR}
WORKDIR ${PROJ_DIR}

# Install Talairach Deamon, SDM, and some Python packages
ENV URL_TD=http://www.talairach.org/talairach.jar \
    URL_SDM=https://www.sdmproject.com/software/updates/SdmPsiGui-linux64-v6.21.tar.gz \
    PATH=${PROJ_DIR}/software/SdmPsiGui-linux64-v6.21:${PATH}
RUN pip install jupytext==1.10.2 duecredit==0.8.1 \
    && apt-get install -y wget fonts-freefont-ttf \
    && mkdir software/ \
    && cd software/ \
    && wget ${URL_TD} \
    && wget ${URL_SDM} \
    && tar -xf SdmPsiGui-linux64-v6.21.tar.gz \
    && rm SdmPsiGui-linux64-v6.21.tar.gz \
    && cd ${PROJ_DIR}

# Install Tini (which is needed to run a jupyter server inside the container)
ENV TINI_VERSION=v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# Change user (see https://mybinder.readthedocs.io/en/latest/tutorials/dockerfile.html)
RUN chown -R ${NB_USER} .
USER ${NB_USER}

# Copy code and data into the container
RUN mkdir code/ && mkdir data/
COPY code/ code/
COPY data/ data/

# Start a jupyter server
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
