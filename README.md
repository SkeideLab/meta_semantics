# A meta-analysis of fMRI studies of semantic cognition in children

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/alexenge/meta_semantics/HEAD)

![meta-analytic results from ALE](https://raw.githubusercontent.com/alexenge/meta_semantics/main/misc/ale_brains.png)

This repository contains the code, data, and results for the following paper:

*APA style citation for the paper/preprint goes here*

## What you can do here:

### 1. Browse through a static version of the project

In the respective folders, you can view the data that went into the meta-analysis, the Python code that was used to perform it, and the results (statistical maps, tables, and figures).

### 2. Reproduce the analyses interactively

If you want rerun or modify any of the code, you have three different options:

#### a) Running the code on a cloud server

Simply hit the "launch Binder" badge at the top or [this link](https://mybinder.org/v2/gh/alexenge/meta_semantics/HEAD) to open an interactive version of the repository on a cloud server (kindly provided by the [Binder project](https://mybinder.readthedocs.io/en/latest/about/about.html)). Please note that launching the server may take a couple of minutes and that the computational resources (CPU cores and memory) are limited.

#### b) Running the code in a local container

If you want to run the code on your local computer, we suggest you do so using our [Docker container](https://hub.docker.com/repository/docker/alexenge/meta_semantics). This will create a small, Linux-based virtual machine which already has all software packages installed. To do so, please first download and install [Docker Desktop](https://www.docker.com/products/docker-desktop). Once it is running, open a command line window (called "Terminal" on Mac and Linux and "PowerShell" or "Git Bash" on Windows). From there, execute the following one-liner:

```
docker run --rm -p 8888:8888 alexenge/meta_semantics:latest
```

You will see a couple of URLs, the last one of which you need to copy and paste into the search bar of your web browser. From there, you will be able to access, execute, and modify our Python notebooks interactively.

Note that [Singularity](https://sylabs.io/singularity/) is an open source alterantive to Docker which will allow you to run the container just as well and even on systems where you do not have root user priveleges. 

#### c) Running the code in your local Python environment

We recommend using one of the two containerized solutions above because they will ensure that you are executing the code in the exact same software environment which we have also used for our paper. However, you can also execute the code directly on your local system.

To do so, please follow these three steps:

1. Make sure you have a recent version (> 3.6) of Python installed. You can test this by typing `python3 --version` on the command line. If you do not yet have Python installed, we recommend doing so via the [Anaconda](https://www.anaconda.com/products/individual) toolkit.

2. Clone the repository from GitHub, either by (a) [downloading it as a zip file](https://github.com/alexenge/meta_semantics/archive/refs/heads/main.zip) or by (b) opening the command line and typing `git pull https://github.com/alexenge/meta_semantics.git`. Note that the latter requires you to have a local installation of [git](https://git-scm.com/downloads).

3. Install the necessary Python packages by opening a new command line, navigating into the directory that you have just downloaded (e.g., `cd meta_semantics`), and executing the following command: `pip3 install -U -r requirements.txt`.

Then you can use your IDE of choice (e.g., the Spyder IDE shipped with Anaconda) to open and run our Python scripts locally.

### 3. Contact us

We are glad to receive any feedback, questions, or criticisms on this project. Simply [open an issue on GitHub](https://github.com/alexenge/meta_semantics/issues/new/choose) or use the corresponding author's e-mail address as provided in the paper.

Thanks a lot for your time and interest.

![SkeideLab and MPI CBS logos](https://raw.githubusercontent.com/alexenge/meta_semantics/main/misc/header_logos.png)
