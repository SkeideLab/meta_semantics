FROM jupyter/minimal-notebook:8d32a5208ca1

# Set some environment variables
ENV PROJDIR=${HOME}/mask_children \
    SDM_PREFIX=SdmPsiGui-linux64-v6.21
    SDM_URL=https://www.sdmproject.com/software/updates/${SDM_PREFIX}.tar.gz \
    PATH=${HOME}/${SDM_PREFIX}:${PATH}

# Install SDM and Python packages
COPY requirements.txt .
RUN wget ${SDM_URL} && \
    tar -xf ${SDM_PREFIX}.tar.gz && \
    chmod -R o+rX ${SDM_PREFIX} && \
    pip install -U -r requirements.txt

# Copy code and data into the container
WORKDIR ${PROJDIR}
COPY code/ code/
COPY data/ data/

# Add permissions for default user
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}
