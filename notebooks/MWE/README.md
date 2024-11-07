# Minimal Working Examples

The Jupyter notebooks in this directory provide initial working examples and minimal code needed to install and run each foundation model on the Great Lakes computing cluster. Model weights are stored in the `umms-indikar/shared/projected/DARPAI_AI/` directory, so as long as you have access to this location, all that is required to run these notebooks is a GPU session and the specific dependencies for each model.

*Note*: Most models use standard dependencies, so setting up dedicated environments was generally unnecessary. However, for scGPT, I created a dedicated Conda environment due to its more complex setup requirements.
