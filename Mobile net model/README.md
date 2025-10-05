# Malaria Diagnosis CNN Transfer Learning

## Project Overview

This repository contains a comprehensive implementation of convolutional neural networks for automated malaria diagnosis using microscopic cell images. The project demonstrates transfer learning techniques with different models.

## Problem Statement

Malaria remains a critical global health challenge with over 219 million cases and 430,000 deaths annually. Traditional microscopic diagnosis is time-consuming, resource-intensive, and prone to human error, particularly in resource-constrained regions. This project applies deep learning to automate malaria diagnosis from blood smear images, potentially improving accuracy and accessibility of diagnosis.

## Dataset

**Source**: NIH Malaria Dataset (National Library of Medicine)
- **Total Images**: ~27,558 cell images
- **Classes**: Binary classification (Parasitized vs Uninfected)
- **Format**: PNG images of blood cells
- **Resolution**: Variable, standardized to 224x224

The dataset contains microscopic images of blood cells collected from patients, with expert annotations distinguishing between parasitized cells (containing malaria parasites) and uninfected cells.


**Key Features**:
- Depthwise separable convolutions
- Reduced parameter count compared to traditional CNNs
- Optimized for mobile deployment
- Pre-trained on ImageNet

### Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download dataset (automated in notebook)
4. Run Jupyter notebook

## Group Project Context

This MobileNet implementation represents one component of a five-model comparative study:
1. Baseline CNN (collective effort)
2. Advanced CNN with enhancements
3. Four pre-trained models (VGG16, ResNet, MobileNet, etc.)

Each model undergoes rigorous evaluation with two experiments, comprehensive metrics reporting, and detailed error analysis to establish performance rankings and clinical applicability.

## Limitations and Future Work

### Current Limitations
- Dataset bias toward specific microscopy conditions
- Limited ethnic and geographic diversity
- Single magnification level
- Binary classification only


## Authors

Formative Group 2- Deep Learning for Medical Diagnosis
- Team collaboration on baseline CNN
- Individual ownership of specialized models
- Unified methodology and evaluation framework



