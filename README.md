# Brain_tumor_classifier 🧠

### Classifying Brain Tumors from MRI Images

Hello! 👋 Welcome to my deep learning project, where I'm building a model to classify brain tumors using MRI images. This project delves into how artificial intelligence can assist in identifying different types of brain tumors with high accuracy, helping healthcare professionals in diagnosing and treating patients faster and more effectively.

The entire pipeline is designed to be reproducible and scalable, so you can easily follow along and replicate the results on your local machine or in the cloud. The goal is to help doctors and radiologists make quicker, more informed decisions, ultimately improving patient care and outcomes.

## 📝 **Problem Description**

Brain tumors are abnormal growths of tissue in the brain, which can be either benign or malignant.

### **Objective**

The main objective of this project is to develop a deep learning model that classifies brain tumors into four categories: **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**. The model is trained using MRI images and aims to assist in the automatic detection and classification of these tumors.

### 📊 **Dataset**

This project uses a combination of datasets sourced from Kaggle and Figshare, which contains 7023 labeled MRI images of human brains classified into four categories:

- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary**

For more detailed information about the dataset, including image formats and any preprocessing steps, please refer to the [data folder](./data/README.md).

Ready to dive into the world of AI and healthcare? Let’s get started and see how we can help detect brain tumors more accurately! 🌟

## 🔧 Tools & Techniques

To bring this project to life, I used:

- **Containerization:** Docker and Docker Compose
- **Web Application Framework (Local Deployment):** Flask (for local web deployment)
- **Web Application Framework (Cloud Deployment):** Streamlit (for cloud-based web deployment)

## ✨ Setup

### **Local Setup**

#### **Clone the Repository**:

```bash
git clone https://github.com/Tobai24/Brain_tumor_classifier.git
cd  Brain_tumor_classifier
```

#### **Set Up the Python Environment**:

**Option 1: Using `pipenv`** (Recommended)

- Install Pipenv using your system's package manager (preferred for Debian-based systems):

  ```bash
  sudo apt install pipenv
  ```

  Alternatively, you can install Pipenv via `pip`:

  ```bash
  pip install pipenv
  ```

- Install the dependencies with `pipenv`:

  ```bash
  pipenv install
  ```

- Activate the `pipenv` shell:
  ```bash
  pipenv shell
  ```

**Option 2: Using `requirements.txt`** (For users preferring `pip`)

- Create and activate a virtual environment:

  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows use: venv\Scripts\activate
  ```

- Install the dependencies:
  ```bash
  pip install -r requirements.txt
  ```

---

### 📝 Notes:

- If you use `pipenv`, you do not need to install the `requirements.txt` dependencies manually because `pipenv` reads the `Pipfile` and manages the environment for you.
- For Debian-based systems, using `sudo apt install pipenv` ensures compatibility with the system Python environment and avoids issues with the "externally managed environment" restriction.

## Exploratory Data Analysis and Modeling

The exploratory data analysis and modeling are done in the [notebooks directory](notebooks/). The exploratory data analysis and model building are done in the `notebook.ipynb` notebook.

The notebook directory also contains the model called `model.pkl`, where the model from the `notebook.ipynb` is stored.

It also contains the training script (which contains the script for training the model with the best AUC) which you can run by running `python train.py` in the terminal

```bash
python train.py
```

## Get Going

Ready to dive into your project? Here’s a quick guide to get you started.

### 📁 **Deployment**

### **Local Deployment**

- **Tools Used**: Flask for building your web app and Docker for containerizing it.
- **Where to Find It**: Head over to the [deployment/local_deployment](deployment/local_deployment) folder.

The README in that folder covers everything you need to get your app running locally.

It’s got the details for setting up Flask and Docker, so you can test things out on your own machine.

### **Cloud Deployment**

- **Tools Used**: Streamlit community cloud for hosting your app and Streamlit for the web interface.
- **Where to Find It**: Navigate to the [deployment/web_deployment](deployment/web_deployment) folder.

The README in that folder guides you through deploying your app using Streamlit. It’s perfect for getting your app live on the cloud.

## 🎉 Special Mentions

A huge thank you to [DataTalks.Club](https://datatalks.club) for offering their ML course for free! 🌟 The insights and skills I gained from their course were crucial in developing this project.

If you want to enhance your expertise in machine learning, I highly recommend checking out their [course](https://github.com/DataTalksClub/machine-learning-zoomcamp). It’s an invaluable resource for diving into machine learning! 📚✨
