Source code for submissions DiMP, SuperDiMP, and LWTL

1) Run the installation script to install all the dependencies. You need to provide the conda install path (e.g. ~/anaconda3) and the name for the created conda environment (here vot2020-pytracking).

bash install.sh conda_install_path vot2020-pytracking

The installation script will download all the networks. In case of issues, please follow the instructions in INSTALL.md to manually install the dependencies.

2) If the networks are not downloaded automatically, download them manually from https://drive.google.com/open?id=1FzGPxJy2k1rrY8bYGR6z20TySJVCACuI and save them with the same names (i.e. dimp50.pth.tar, super_dimp.pth.tar, lwtl.pth.tar) in the folder pytracking/networks.

3) Copy the configuration file trackers.ini to the vot-workspace. Set the path to the source code (line 8, 18, and 28).

4) The trackers can be evaluated as 

vot evaluate --workspace ABSOLUTE_PATH_TO_VOT_WORKSPACE LWTL
vot evaluate --workspace ABSOLUTE_PATH_TO_VOT_WORKSPACE DiMP
vot evaluate --workspace ABSOLUTE_PATH_TO_VOT_WORKSPACE SuperDiMP

Note: The evaluation might crash with timeout when run for the first time due to the time taken for compiling the PreciseROIPooling module. Re-run the evaluation in this case, or run with a longer timeout value.
