## Notebook Directory

This directory contains all the resources related to **exploratory data analysis (EDA)**, **data preprocessing**, **model training**, and **performance evaluation** for the brain tumor classification project.

## 📝 **Files Overview**

### **`notebook.ipynb`**

This Jupyter notebook contains the complete workflow for the project, including:

- **Exploratory Data Analysis (EDA)**: Visualizing the dataset, checking for class imbalances, and understanding image characteristics.
- **Data Preprocessing**: Resizing images, normalizing pixel values, and splitting the dataset into training, validation, and test sets.
- **Model Training**: Designing and training a convolutional neural network (CNN) using PyTorch.
- **Model Evaluation**: Comparing the performance of different models based on accuracy.

### **`train.py`**

This Python script automates the training process for the best-performing model. It:

- Loads the preprocessed dataset.
- Trains the model using the best hyperparameters.
- Saves the trained model to the `best_models/` folder.

Run the script using:

```bash
python train.py
```

### **`models/`**

This folder contains the top 3 trained models saved as `.pth` files. These models are selected based on their performance on the validation set.

### **`performance/`**

This folder contains CSV files with detailed performance metrics for each of the top 3 models. Metrics include:

- Accuracy

## **Getting Started**

1. Open the `notebook.ipynb` file to explore the dataset and understand the model training process.
2. Use the `train.py` script to train the best model and save it for deployment.
3. Evaluate the models using the performance metrics stored in the `performance/` folder.
