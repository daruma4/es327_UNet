# es327Project_UNet
<img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License"/>

Implements the U-Net convolutunal neural network to segment vessels in ultrasound (US) scans of the arm.
<p align="center">
  <img src="https://github.com/daruma4/es327_UNet/blob/main/assets/results.png?raw=true" alt="Vessel Segmentation Sample"/>
</p>

## Installation

Best achieved using an Anacodna environment: https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html
1. Setup conda environment using
   ```
   Python=3.7.11
   ```
2. In conda environment run
   ```
   conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
   ```
3. Clone this repository
4. Install dependencies
   ```
   pip install -r requirements.txt
   ```
