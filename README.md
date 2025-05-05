# DiffPIE
This repository demonstrates two examples of using DiffPIE to generate protein structures under realistic external environments: Aβ(16–22) peptide adsorption on an Au(111) surface and a stapled peptide (PDB: 8Q1R)

Each case involves two main tasks:
Biasing force generation and application during Str2Str inference
Sidechain generation and system construction

Example 1: Aβ(16–22) Adsorption on Au(111)
1. Biasing Force Setup and Inference
The code for generating biasing forces is located in:

    Str2Str_inference/Abeta_16-22/GoldP/kTlogP.py

To run inference with biasing: Place the GoldP folder and Abeta_biasing.py script into the root of your local Str2Str codebase. Replace the following files in Str2Str/src/ with the modified versions provided: diffusion_module.py, frame.py, and r3.py. Use the included diffusion.yaml as the recommended configuration for inference. Feel free to modify settings based on your system.

2. Sidechain Generation and System Construction
After inference, use the post-processing script:

    Structure_generations/Abeta/DiffPIE_sidechain_Au.py

This script integrates sidechains using MD simulation data. Required MD trajectories for this step can be downloaded from:[https://drive.google.com/drive/folders/1dFnwRA-LjiuNRPm_x5LmsAWaiW4Khl0h?usp=drive_link
](https://drive.google.com/drive/folders/1nnBrfQ27QTpYfI27mRsjAwOfXVl-vZak?usp=drive_link)

Example 2: Stapled Peptide (PDB: 8Q1R)
1. Biasing Force Setup and Inference
The code for generating biasing forces is located in:

    Str2Str_inference/P3_F/logP_cp.py
   
To run inference with biasing: Place the gmm_potential_model.pkl script into the root of your local Str2Str codebase. Replace the following files in Str2Str/src/ with the modified versions provided: diffusion_module.py, frame.py, and r3.py. Use the included diffusion.yaml as the recommended configuration for inference. Feel free to modify settings based on your system.


2. Sidechain Generation and System Construction
After inference, use the post-processing script:

    SStructure_generations/P3-F

This script integrates sidechains using MD simulation data. Required MD trajectories for this step can be downloaded from:[https://drive.google.com/drive/folders/1dFnwRA-LjiuNRPm_x5LmsAWaiW4Khl0h?usp=drive_link](https://drive.google.com/drive/folders/118ouP09jWunvZtXFiRzfB03JBGx3UkYj?usp=drive_link)

