FROM neurostuff/nimare:latest

# Install xdg-open which is needed for SDM
RUN add-apt-repository -r ppa:openjdk-r/ppa && \
    apt update -q && \
    apt-get install -y xdg-utils

# Update IPython so we can execute the notebooks
RUN conda install -n nimare -y ipython==7.2.0
