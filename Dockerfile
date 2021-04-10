FROM jupyter/minimal-notebook:8d32a5208ca1

# Install SDM and Python packages
COPY requirements.txt .
ENV URL_SDM=https://www.sdmproject.com/software/updates/SdmPsiGui-linux64-v6.21.tar.gz \
    PATH=${HOME}/software/SdmPsiGui-linux64-v6.21:${PATH}
RUN wget -P software/ ${URL_SDM} && \
    tar -xf software/SdmPsiGui-linux64-v6.21.tar.gz -C software/ && \
    pip install -U -r requirements.txt

# Copy code and data into the container
COPY code/ code/
COPY data/ data/

# Add permissions for default user
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}
