# ECG Arrhythmia Detector

This repository contains a complete pipeline for classifying heart arrhythmias from raw ECG signals using the MIT-BIH Arrhythmia Database.

## Overview
This project trains a 1D Convolutional Neural Network (CNN) to classify ECG beats into one of five standard AAMI superclasses:
- **N**: Normal beat
- **S**: Supraventricular ectopic beat
- **V**: Ventricular ectopic beat
- **F**: Fusion beat
- **Q**: Unknown beat

## Evaluation Methodology
To avoid data leakage and ensure realistic performance estimates, this project uses a **patient-independent split** (also known as inter-patient evaluation).
- **Train Set (DS1)**: Records 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230
- **Test Set (DS2)**: Records 100, 103, 104, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 217, 219, 221, 222, 228, 231, 232, 233, 234

## Getting Started

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the Data**
   Extract the MIT-BIH dataset into `mit-bih-arrhythmia-database-1.0.0/`.
   Run the data processing script to extract beats and split the data:
   ```bash
   python src/data/make_dataset.py
   ```
   This will generate `.npy` files in `data/processed/`.

3. **Train the Model**
   Run the training script to train the 1D-CNN and evaluate its performance on the test set:
   ```bash
   python src/models/train_model.py
   ```
