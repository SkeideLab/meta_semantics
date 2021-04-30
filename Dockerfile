FROM jupyter/minimal-notebook:8d32a5208ca1

# Set some environment variables
ENV PROJDIR=${HOME}/meta_semantics \
    SDM_PREFIX=SdmPsiGui-linux64-v6.21
ENV SDM_URL=https://www.sdmproject.com/software/updates/${SDM_PREFIX}.tar.gz \
    PATH=${HOME}/${SDM_PREFIX}:${PATH}

# Install Python packages and SDM
COPY requirements.txt .
RUN pip install -U -r requirements.txt && \
    wget ${SDM_URL} && \
    tar -xf ${SDM_PREFIX}.tar.gz && \
    chmod -R o+rX ${SDM_PREFIX}

# Copy directories into the container
WORKDIR ${PROJDIR}
COPY code/ code/
COPY data/ data/
COPY misc/ misc/
COPY results/ results/

# Add permissions for default user
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}
