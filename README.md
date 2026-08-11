# ⚙️ Auto ML Preprocessor

An automated machine learning preprocessing tool designed to reduce the repetitive work involved in **EDA, data preprocessing, feature engineering, scaling, and feature selection**.

The project supports both **supervised** and **unsupervised** learning workflows and provides a Streamlit interface for uploading datasets, analyzing them, processing them, previewing the results, and downloading the generated outputs.

> **No more manual EDA. No more repetitive preprocessing.**

---

## 🚀 Live Demo

Try the deployed application:

**[👉 Click here to try the deployed app](https://automated-ml-preprocessing.streamlit.app/)**

---

## 📓 Jupyter / Colab Notebooks

### Supervised Learning

The supervised preprocessing workflow handles datasets containing a target variable and prepares the data for downstream machine learning models.

**[👉 Click here to open the Supervised Learning notebook](https://drive.google.com/file/d/1U4UVvxlxXWTV5SKuxUw5U2xMEEPFOaUj/view?usp=sharing)**

### Unsupervised Learning

The unsupervised workflow is designed for datasets without a target variable and follows a separate preprocessing path.

**[👉 Click here to open the Unsupervised Learning notebook](https://drive.google.com/file/d/1PrboXqurh9mhsEFj40RUuVoMEUKp9hTx/view?usp=sharing)**

---

## ✨ Features

### 🧠 Learning Type Selection

The application supports two processing modes:

- **Supervised Learning**
  - Works with a target variable.
  - Automatically detects the type of supervised task.
  - Keeps the target separate from feature preprocessing.

- **Unsupervised Learning**
  - Does not require a target variable.
  - Processes the feature matrix directly.
  - Uses a separate target-free preprocessing pipeline.

---

### 📂 Dataset Workflows

For both supervised and unsupervised processing, the application supports:

#### Entire Dataset

Upload a complete dataset and automatically split it into training and testing data.

The test size can be selected by the user.

**Default: 20% test / 80% training**

#### Training Dataset

Upload an already-prepared training dataset.

No additional train/test split is performed.

#### Test Dataset

Upload separate training and test datasets.

The preprocessing pipeline is:

1. Fitted using the training dataset.
2. Learned preprocessing parameters are retained.
3. The same transformations are applied to the test dataset.

This prevents information from the test dataset from leaking into the training preprocessing stage.

---

## 📊 Automated EDA

The application performs automated exploratory data analysis before preprocessing.

The analysis includes information such as:

- Dataset dimensions
- Feature types
- Missing-value information
- Unique-value counts
- Numerical feature statistics
- Distributions
- Skewness
- Outlier visualization
- Categorical feature information
- Numerical feature visualizations
- Box plots

The project also uses **Sweetviz** for automated exploratory analysis.

---

## 🧹 Automated Preprocessing

Depending on the dataset, the preprocessing pipeline can handle common preprocessing tasks such as:

- Missing-value handling
- Numerical feature processing
- Categorical feature processing
- Encoding categorical variables
- Skewed numerical features
- Feature scaling
- Feature selection
- ID-like column handling

The exact preprocessing path depends on the structure of the uploaded dataset.

---

## 🔬 Supervised Processing Pipeline

The supervised workflow follows a target-aware preprocessing path.

A typical workflow is:

```text
Dataset
   ↓
Target Identification
   ↓
EDA
   ↓
Train / Test Split
   ↓
Missing Value Handling
   ↓
Categorical Feature Processing
   ↓
Numerical Feature Processing
   ↓
Feature Engineering
   ↓
Scaling
   ↓
Feature Selection
   ↓
Processed X_train / X_test
```

When separate training and test datasets are provided, the pipeline is fitted only on the training data before being applied to the test data.

---

## 🔬 Unsupervised Processing Pipeline

The unsupervised workflow does not use a target variable.

Its general flow is:

```text
Dataset
   ↓
EDA
   ↓
Train / Test Split (if required)
   ↓
ID-like Column Detection
   ↓
Missing Value Handling
   ↓
Categorical Feature Processing
   ↓
Skewed Feature Transformation
   ↓
Scaling
   ↓
Processed Feature Matrix
```

The unsupervised pipeline is intentionally separate from the supervised pipeline because the absence of a target changes several preprocessing decisions.

---

## 📦 Output

Processed datasets are generated as downloadable files.

Depending on the selected workflow, the output can contain:

- `X_train.csv`
- `X_test.csv`
- Pipeline information
- Complete downloadable package

The generated outputs are timestamped so that different preprocessing runs can be distinguished.

---

## 🖥️ Application Workflow

The overall application workflow is:

```text
📁 Upload
   ↓
🧠 Choose Learning Type
   ↓
📊 Automated EDA
   ↓
⚙️ Automated Preprocessing
   ↓
🎯 Feature Selection
   ↓
🔍 Processed Dataset Preview
   ↓
📥 Download
```

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- SciPy
- Scikit-learn
- Sweetviz
- Plotly
- FastAPI
- Uvicorn
- Streamlit
- Requests

---

## 📁 Project Structure

A simplified structure of the project is:

```text
Data_Preprocessing/
│
├── app.py
├── main.py
│
├── supervised_pipeline.py
├── unsupervised_pipeline.py
│
├── Data_Preprocessing.ipynb
│
├── outputs/
│
├── requirements.txt
└── README.md
```

The project uses separate preprocessing pipelines for supervised and unsupervised learning to keep the two workflows independent and easier to maintain.

---

## ⚙️ Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/Koustav-2003/automated-ml-preprocessing.git
cd automated-ml-preprocessing
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the FastAPI backend

```bash
uvicorn main:app --reload
```

### 4. Start the Streamlit application

```bash
streamlit run app.py
```

The Streamlit frontend communicates with the FastAPI backend to perform the preprocessing operations.

---

## 📌 Important Notes

- The **Entire Dataset** workflow uses a configurable test split with **20% as the default**.
- The **Training Dataset** workflow does not perform another train/test split.
- The **Test Dataset** workflow expects separate training and test datasets.
- For supervised learning, the target variable is not treated as a normal input feature during feature preprocessing.
- For unsupervised learning, no target variable is required.
- When separate train/test datasets are supplied, preprocessing is fitted on the training data and then applied to the test data.

---

## 🎯 Purpose of the Project

This project was built primarily as a practical **machine learning automation tool** to reduce repetitive preprocessing work.

It is not intended to replace understanding of EDA or preprocessing. Instead, the goal is to automate the repetitive implementation so that more time can be spent on:

- Understanding the dataset
- Selecting appropriate models
- Feature engineering decisions
- Model evaluation
- Experimentation

It also serves as a practical implementation of common preprocessing techniques learned throughout the machine learning workflow.

---

## 🔮 Future Improvements

Possible future versions could extend the tool with:

- Automated model recommendation
- Automated supervised model training
- Hyperparameter tuning
- Clustering model selection
- Dimensionality reduction
- Automated anomaly detection
- More feature-engineering strategies
- Model comparison
- Explainable AI
- Exportable preprocessing pipelines
- MLOps integration

---

## 👨‍💻 Project

**Auto ML Preprocessor**

Built as a practical machine learning automation project to streamline the repetitive parts of the ML workflow.

**Live application:**  
[👉 Click here to try the deployed app](https://automated-ml-preprocessing.streamlit.app/)

**Supervised notebook:**  
[👉 Click here to open](https://drive.google.com/file/d/1U4UVvxlxXWTV5SKuxUw5U2xMEEPFOaUj/view?usp=sharing)

**Unsupervised notebook:**  
[👉 Click here to open](https://drive.google.com/file/d/1PrboXqurh9mhsEFj40RUuVoMEUKp9hTx/view?usp=sharing)
