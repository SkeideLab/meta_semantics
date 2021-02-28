FROM neurostuff/nimare:latest

# Activate the conda environment
ENV PATH=/opt/miniconda-latest/envs/nimare/bin:$PATH \
    CONDA_DEFAULT_ENV=nimare \
    CONDA_PREFIX=/opt/miniconda-latest/envs/nimare

# Install current versions of IPython and jupytext
RUN conda install -y -c conda-forge ipython==7.2.0 jupytext==1.10.2

# Copy code and data into the container
RUN mkdir -p /home/workspaces/mask_children/code/ && \
    mkdir /home/workspaces/mask_children/data/
WORKDIR /home/workspaces/mask_children
COPY code/ code/
COPY data/ data/

# Add Tini (which is needed to run a jupyter server in the container)
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

# Start a jupyter server
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
