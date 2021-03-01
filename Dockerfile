FROM neurostuff/nimare:latest

# Activate the conda environment
ENV PATH=/opt/miniconda-latest/envs/nimare/bin:$PATH \
    CONDA_DEFAULT_ENV=nimare \
    CONDA_PREFIX=/opt/miniconda-latest/envs/nimare

# Install IPython, jupytext, and SDM (https://www.sdmproject.com/)
RUN conda install -y -c conda-forge ipython==7.2.0 jupytext==1.10.2 \
    && apt-get install -y wget \
    && wget https://www.sdmproject.com/software/updates/SdmPsiGui-linux64-v6.21.tar.gz \
    && mkdir -p /home/workspaces/mask_children/software/ \
    && tar -xf SdmPsiGui-linux64-v6.21.tar.gz -C /home/workspaces/mask_children/software/ \
    && rm SdmPsiGui-linux64-v6.21.tar.gz

# Install Tini (which is needed to run a jupyter server inside the container)
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# Set working directory
WORKDIR /home/workspaces/mask_children/

# Change user and give them the necessary permissions
# (see https://mybinder.readthedocs.io/en/latest/tutorials/dockerfile.html)
ARG NB_USER=neuro
RUN chown -R ${NB_USER} .
USER ${NB_USER}

# Copy code and data into the container
RUN mkdir code/ && mkdir data/
COPY code/ code/
COPY data/ data/

# Start a jupyter server
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
