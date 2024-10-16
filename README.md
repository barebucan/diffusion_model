# Diffusion Model Repository
This repository contains an implementation of a simple diffusion model designed for generating images based on the MNIST dataset. The project includes a UNet architecture and a classifier-guided sampling method, providing a comprehensive approach to the diffusion process.

## Project Structure
The repository consists of three main files, each fulfilling a specific role in the overall diffusion model:

### 1. `main.py`
This file implements a simple diffusion model on the MNIST dataset. Key features include:
- **Label Embedding**: Incorporates label embeddings to condition the diffusion process based on the digit label.
- **Diffusion Process**: Utilizes a diffusion model to iteratively generate samples from random noise.

### 2. `UNet.py`
This file contains the implementation of the UNet architecture, which serves as the core network for the diffusion model. The UNet includes:
- **Down-sampling and Up-sampling Layers**: Facilitates effective feature extraction and image reconstruction.
- **Skip Connections**: Ensures effective information flow between layers for improved performance.
- **Flexibility**: Adaptable to meet the specific needs of the diffusion process.

### 3. `Classifier_Guidance.py`
This script implements a diffusion model with classifier guidance, trained on images with dimensions of 3x96x96. Key details include:
- **Classifier-Guided Sampling**: Enhances the quality of generated samples by integrating class-specific gradients from a pre-trained classifier.
- **Training Resolution**: The model is specifically trained on images with a resolution of 3x96x96, allowing for high-quality image generation.

## Example Outputs

### Number Transformation
![Number Transformation](number_transform.png)
This image demonstrates the transformation of one number into another using our diffusion model.

### Number Generation from Noise
![Number Generation](number_generated.png)
This image showcases a number generated from random noise using our diffusion model.
