# Chest Cancer Classification using MLflow

This project is an end-to-end machine learning application that classifies chest cancer from CT scan images. It leverages deep learning techniques, with the entire machine learning lifecycle tracked and managed using MLflow. The project includes a web interface for making predictions and serves as a practical implementation of machine learning in healthcare.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Directory Structure](#directory-structure)
4. [Setup Instructions](#setup-instructions)
5. [Usage](#usage)
6. [Model Training](#model-training)
7. [Evaluation Metrics](#evaluation-metrics)


## Project Overview
Chest cancer is one of the most prevalent and deadliest forms of cancer. Early detection through imaging techniques such as CT scans plays a crucial role in treatment. This project uses deep learning models to classify whether a CT scan shows signs of chest cancer.

Key features of this project include:
- **Model Training**: Deep learning model trained on chest CT scan images.
- **MLflow Integration**: MLflow is used to track experiments, log metrics, and manage model versions.
- **Web Interface**: A simple web interface that allows users to upload CT scan images and get predictions.

## Technologies Used
- **Python**: The main programming language used for model development and backend logic.
- **Jupyter Notebooks**: Used for data analysis, model training, and experimentation.
- **MLflow**: For tracking machine learning experiments and managing model versions.
- **TensorFlow/PyTorch**: Used for implementing the deep learning model.
- **Flask**: Backend web framework used to create the prediction API and web interface.
- **HTML/CSS**: For building the frontend interface.


## Directory Structure

The repository is organized as follows:


Chest-Cancer-Classification-MLflow/
│
├── .github/workflows/               # GitHub Actions for CI/CD (if configured)
├── config/                          # Configuration files for model training
├── research/                        # Jupyter notebooks and research scripts
├── src/chest_cancer_classifier/     # Source code for the chest cancer classifier
├── templates/                       # HTML templates for the frontend
│
├── .gitignore                       # Ignore specific files in git
├── LICENSE                          # Project license
├── README.md                        # This file
├── app.py                           # Flask application for predictions
├── inputImage.jpg                   # Example input image for predictions
├── main.py                          # Main script for running the web server
├── params.yaml                      # Model hyperparameters and configuration
├── requirements.txt                 # Project dependencies
├── scores.json                      # Evaluation metrics
├── setup.py                         # Script to package the project
└── template.py                      # Python script template (if needed)



## Setup Instructions

### Prerequisites
- **Python 3.7+**: Make sure Python is installed.
- **Pip**: Ensure `pip` is installed to manage Python packages.

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/aditi-singh-21/Chest-Cancer-ML-OPS.git
    cd Chest-Cancer-ML-OPS
    ```

2. Set up a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up MLflow (optional):
   - Install MLflow:
     ```bash
     pip install mlflow
     ```
   - Start MLflow tracking server (if using locally):
     ```bash
     mlflow ui
     ```

5. Run the application:
    ```bash
    python app.py
    ```

## Usage
1. **Web Interface**: After starting the Flask app, navigate to `http://localhost:5000` in your web browser. Upload a CT scan image and click on "Predict" to get the classification result.
   
2. **Command Line Interface**: You can also interact with the model via the command line. Use `main.py` to run predictions on sample images.

## Model Training
To retrain the model:
1. Update the `params.yaml` file with your desired hyperparameters.
2. Use the Jupyter notebooks in the `research/` directory to train the model. Ensure that you have the dataset of chest CT scans available and properly configured in the notebook.
3. The model training process includes:
   - **Data Preprocessing**: Image augmentation and normalization.
   - **Model Architecture**: A deep neural network designed for image classification.
   - **Training & Validation**: The model is trained on labeled CT scan data, and performance is validated on a holdout set.

## Evaluation Metrics
After training, evaluation metrics are stored in `scores.json`. Key metrics include:
- **Accuracy**


These metrics help in assessing the model's performance and can be visualized in the Jupyter notebooks under the `research/` folder.




