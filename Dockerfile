FROM neurostuff/nimare:latest

# Update IPython so we can execute jupyter notebooks
RUN conda install -n nimare -y ipython==7.2.0
