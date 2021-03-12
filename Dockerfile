FROM neurostuff/nimare:latest

# Activate the conda environment
ENV PATH=/opt/miniconda-latest/envs/nimare/bin:$PATH \
    CONDA_DEFAULT_ENV=nimare \
    CONDA_PREFIX=/opt/miniconda-latest/envs/nimare

# Set working directory
RUN mkdir -p /workspaces/mask_children/
WORKDIR /workspaces/mask_children/

# Install IPython, jupytext, Talairach Deamon, and SDM
ENV URL_TD http://www.talairach.org/talairach.jar
ENV URL_SDM https://www.sdmproject.com/software/updates/SdmPsiGui-linux64-v6.21.tar.gz
RUN pip install jupytext==1.10.2 duecredit==0.8.1 \
    && apt-get install -y wget \
    && mkdir -p software/ \
    && wget -P software/ ${URL_TD} \
    && wget -P software/ ${URL_SDM} \
    && tar -xf software/SdmPsiGui-linux64-v6.21.tar.gz -C software/ \
    && rm software/SdmPsiGui-linux64-v6.21.tar.gz

# Install Tini (which is needed to run a jupyter server inside the container)
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# Change user and give them the necessary permissions
# (see https://mybinder.readthedocs.io/en/latest/tutorials/dockerfile.html)
ENV NB_USER=neuro
RUN chown -R ${NB_USER} .
USER ${NB_USER}

# Copy code and data into the container
RUN mkdir code/ && mkdir data/
COPY code/ code/
COPY data/ data/

# Start a jupyter server
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
