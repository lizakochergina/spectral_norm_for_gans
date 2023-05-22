# Explore Adaptive Spectral Normalization for Generative Models
The purpose of this project is to explore usage of spectral normalization and its variations in generative model for data simulation of the Time Projection Chamber tracker of
the MPD experiment at the NICA accelerator complex.

At this point current model supports application of standart spectral normalization for both generator and discriminator.

To run experiment you need to adjust configs in init.py file and then run `python main.py` from current directory.

You can see metric results of completed experiments in the folder `metrics`. 

Data, module trends and some parts of module evaluation are brought from repo https://github.com/SiLiKhon/TPC-FastSim. Model is based on article https://link.springer.com/article/10.1140/epjc/s10052-021-09366-4.
