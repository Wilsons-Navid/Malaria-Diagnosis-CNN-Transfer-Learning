# Malaria Diagnosis CNN Transfer Learning

## Project Overview

This repository contains a comprehensive implementation of convolutional neural networks for automated malaria diagnosis using microscopic cell images. The project demonstrates transfer learning techniques with MobileNet architecture as part of a larger comparative study involving multiple CNN models for medical image classification.

## Problem Statement

Malaria remains a critical global health challenge with over 219 million cases and 430,000 deaths annually. Traditional microscopic diagnosis is time-consuming, resource-intensive, and prone to human error, particularly in resource-constrained regions. This project applies deep learning to automate malaria diagnosis from blood smear images, potentially improving accuracy and accessibility of diagnosis.

## Dataset

**Source**: NIH Malaria Dataset (National Library of Medicine)
- **Total Images**: ~27,558 cell images
- **Classes**: Binary classification (Parasitized vs Uninfected)
- **Format**: PNG images of blood cells
- **Resolution**: Variable, standardized to 224x224 for MobileNet

The dataset contains microscopic images of blood cells collected from patients, with expert annotations distinguishing between parasitized cells (containing malaria parasites) and uninfected cells.

## Model Architecture

### MobileNet Transfer Learning

This implementation focuses on MobileNet, a lightweight CNN architecture designed for mobile and embedded applications. MobileNet utilizes depthwise separable convolutions to reduce computational complexity while maintaining performance.

**Key Features**:
- Depthwise separable convolutions
- Reduced parameter count compared to traditional CNNs
- Optimized for mobile deployment
- Pre-trained on ImageNet

### Experimental Design

Two primary experiments were conducted:

1. **Feature Extraction** (`trainable=False`)
   - Frozen pre-trained weights
   - Only classification head trained
   - Lower computational requirements
   - Reduced overfitting risk

2. **Fine-tuning** (`trainable=True`)
   - All layers trainable
   - Domain-specific adaptation
   - Higher computational cost
   - Potential for better performance

## Methodology

### Data Preprocessing
- Image rescaling (0-1 normalization)
- Data augmentation (rotation, shifts, zoom, horizontal flip)
- Train-validation split (80-20)
- Batch processing (batch size: 32)

### Training Configuration
- **Input Size**: 224x224x3
- **Optimizer**: Adam
- **Loss Function**: Binary crossentropy
- **Metrics**: Accuracy, Precision, Recall
- **Callbacks**: Early stopping, learning rate reduction, model checkpointing

### Evaluation Framework
- **Performance Metrics**: Accuracy, Precision, Recall, F1-score, AUC
- **Visualizations**: Learning curves, confusion matrices, ROC curves
- **Error Analysis**: Overfitting assessment, misclassification patterns

## Results Summary

### Performance Comparison

| Metric | Feature Extraction | Fine-tuning | Improvement |
|--------|-------------------|-------------|-------------|
| Accuracy | 95.2% | 96.8% | +1.7% |
| Precision | 94.8% | 96.5% | +1.8% |
| Recall | 95.6% | 97.1% | +1.6% |
| F1-Score | 95.2% | 96.8% | +1.7% |
| AUC | 0.984 | 0.991 | +0.7% |

### Key Findings

- Fine-tuning achieved superior performance across all metrics
- Both models demonstrated excellent diagnostic capability (>95% accuracy)
- Low false negative rates critical for medical applications
- Computational efficiency suitable for mobile deployment

## Clinical Significance

- **High Sensitivity**: Minimizes missed malaria cases
- **High Specificity**: Reduces false alarms
- **Rapid Diagnosis**: Potential for real-time screening
- **Accessibility**: Mobile-ready architecture for resource-limited settings

## Repository Structure

```
├── README.md
├── Malaria_Diagnosis_CNN_Group6_EvenNumber.ipynb
├── best_mobilenet_feature_extraction.h5
├── best_mobilenet_fine_tuning.h5
└── cell_images/
    ├── Parasitized/
    └── Uninfected/
```

## Requirements

### Dependencies
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- OpenCV

### Hardware Requirements
- GPU recommended for training
- Minimum 8GB RAM
- 2GB storage for dataset

## Usage

### Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download dataset (automated in notebook)
4. Run Jupyter notebook

### Training
The notebook includes automated data download, preprocessing, model creation, training, and evaluation. Both feature extraction and fine-tuning experiments are executed sequentially with comprehensive evaluation.

### Inference
Trained models are saved as `.h5` files and can be loaded for inference on new images.

## Technical Implementation

### Model Architecture Details
- **Base Model**: MobileNet (ImageNet pre-trained)
- **Custom Head**: Global Average Pooling → Dropout → Dense (sigmoid)
- **Input Processing**: MobileNet-specific preprocessing
- **Regularization**: Dropout (0.5), early stopping

### Training Strategy
- **Learning Rates**: 0.001 (feature extraction), 0.0001 (fine-tuning)
- **Batch Size**: 32
- **Epochs**: Up to 50 (early stopping)
- **Validation Split**: 20%

## Group Project Context

This MobileNet implementation represents one component of a five-model comparative study:
1. Baseline CNN (collective effort)
2. Advanced CNN with enhancements
3. Four pre-trained models (VGG16, ResNet, MobileNet, etc.)

Each model undergoes rigorous evaluation with seven experiments, comprehensive metrics reporting, and detailed error analysis to establish performance rankings and clinical applicability.

## Limitations and Future Work

### Current Limitations
- Dataset bias toward specific microscopy conditions
- Limited ethnic and geographic diversity
- Single magnification level
- Binary classification only

### Future Enhancements
- Multi-class classification (parasite species)
- Ensemble methods combining multiple architectures
- Real-time mobile application development
- Integration with clinical workflows

## Authors

Group 6 - Deep Learning for Medical Diagnosis
- Team collaboration on baseline CNN
- Individual ownership of specialized models
- Unified methodology and evaluation framework

## References

- Rajaraman, S., et al. (2018). Pre-trained convolutional neural networks as feature extractors for malaria parasite detection.
- National Institutes of Health Malaria Dataset
- MobileNet Architecture (Howard et al., 2017)
- Transfer Learning in Medical Image Analysis

## License

This project is developed for academic purposes as part of a machine learning course assignment. Dataset usage follows NIH public access guidelines.

---

**Note**: This implementation demonstrates transfer learning principles applied to medical image classification, emphasizing rigorous evaluation, clinical relevance, and computational efficiency for real-world deployment scenarios.