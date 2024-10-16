Diffusion Model Repository
This repository contains the implementation of a simple diffusion model for generating images based on the MNIST dataset, alongside a UNet architecture and a classifier-guided sampling method. The repository is structured into three main files, each performing a specific role in the overall diffusion process.

Project Structure
1. main.py
This file implements a simple diffusion model on the MNIST dataset. Key features include:

Label Embedding: The model incorporates label embeddings to condition the diffusion process based on the digit label.
Diffusion Process: A diffusion model is used to iteratively generate samples from random noise.

2. UNet.py
This file contains the implementation of a UNet architecture, which is used within the diffusion model. The UNet serves as the core network responsible for predicting noise in the reverse diffusion process.

The UNet features:

Down-sampling and up-sampling layers.
Skip connections for effective information flow.
Flexibility to adapt to the specific needs of the diffusion process.
3. Classifier_Guidance.py
This script implements a diffusion model with classifier guidance. It is trained on images with dimensions 3x96x96, guiding the diffusion process based on a pre-trained classifier. Classifier guidance improves sample quality by steering the generation process towards desired classes.

Key details:

Classifier-Guided Sampling: Helps refine generated samples by incorporating class-specific gradients from a classifier.
Training Resolution: The model is trained on images with resolution 3x96x96.
