# ⚙️ Auto ML Preprocessor

Automated data preprocessing and feature engineering for supervised and unsupervised machine learning.

**Upload → Optional EDA → Preprocess → Feature Processing → Download**

## 🚀 Try It Out

**Live Streamlit Application:** [Click here](https://automated-ml-preprocessing.streamlit.app/)

**Supervised EDA Notebook:** [Click here](https://drive.google.com/file/d/1Gh45XjIZmRMz7uWzy8_q7E62CyxnYduW/view?usp=sharing)

**Unsupervised EDA Notebook:** [Click here](https://drive.google.com/file/d/1aKTGNc1MWjLFWFfCirj0hwr3H_c87wNn/view?usp=sharing)

**Supervised Feature Engineering Notebook:** [Click here](https://drive.google.com/file/d/1pj4NgWMw2k3qfMl8UEL-exNZBZkLKErN/view?usp=sharing)

**Unsupervised Feature Engineering Notebook:** [Click here](https://drive.google.com/file/d/1BeqviRUdrkt_z7N2fwcQNTW8FxDJYl4C/view?usp=sharing)

**FastAPI Backend:** [Click here](https://automated-ml-preprocessing-api.onrender.com/)

The Streamlit frontend communicates with the FastAPI backend to perform EDA generation and dataset preprocessing.

## ✨ Features

### Learning modes

- **Supervised Learning**
  - Classification
  - Regression
  - Target selection
  - Supervised target encoding

- **Unsupervised Learning**
  - No target variable required
  - One-hot encoding for categorical variables
  - Unsupervised preprocessing workflow

### Dataset workflows

The application supports three input workflows for both supervised and unsupervised processing:

#### 1. Entire Dataset

Upload one complete CSV.

```text
Complete Dataset
      ↓
Train/Test Split
      ↓
Fit preprocessing on training data
      ↓
Transform test data
      ↓
X_train + X_test
```

#### 2. Training Dataset

Upload only the training dataset.

```text
Training Dataset
      ↓
Fit preprocessing
      ↓
Processed Training Dataset
```

No additional train/test split is performed.

#### 3. Training + Test Dataset

Upload separate training and test datasets.

```text
Training Dataset
      ↓
Fit preprocessing
      ↓
Learned preprocessing parameters
      ↓
Apply to training + test
```

The test dataset is kept unseen while preprocessing parameters are learned.

## 📊 EDA

EDA is **optional** and is completely separate from preprocessing.

The application generates a downloadable HTML EDA report based on the project's EDA notebooks.

- EDA does **not** affect preprocessing.
- EDA does **not** change the processed dataset.
- For train/test workflows, EDA is generated from the **training dataset only**.
- EDA is delivered as a downloadable HTML report rather than being displayed as a large visualization dashboard inside the application.
- Sweetviz is not required.

### Supervised EDA

[📊 View Supervised EDA Notebook](https://drive.google.com/file/d/1Gh45XjIZmRMz7uWzy8_q7E62CyxnYduW/view?usp=sharing)

### Unsupervised EDA

[📊 View Unsupervised EDA Notebook](https://drive.google.com/file/d/1aKTGNc1MWjLFWFfCirj0hwr3H_c87wNn/view?usp=sharing)

## 🧠 Preprocessing Pipeline

### Supervised Pipeline

```text
Raw Data
   ↓
ID Detection
   ↓
Missing Value Handling
   ↓
Missing Indicators
   ↓
Rare Category Handling
   ↓
Skewness Transformation
   ↓
Target Encoding
   ↓
Feature Scaling
   ↓
Processed Data
```

### Unsupervised Pipeline

```text
Raw Data
   ↓
ID Detection
   ↓
Missing Value Handling
   ↓
Missing Indicators
   ↓
Rare Category Handling
   ↓
Skewness Transformation
   ↓
One-Hot Encoding
   ↓
Feature Scaling
   ↓
Processed Data
```

## 🛡️ Data Leakage Prevention

For separate train/test workflows, preprocessing parameters are learned from the training data and then applied to the test data.

```text
Training Data
      ↓
Fit preprocessing
      ↓
Learn:
  • imputation values
  • category mappings
  • scaling parameters
  • other learned transformations
      ↓
Apply learned preprocessing
      ↓
Training Data + Test Data
```

The test dataset is not used to learn preprocessing parameters.

## 🎯 Feature Engineering

Feature engineering is intentionally kept separate from automatic model-based feature selection.

The project does **not** perform automatic Lasso/L1 feature elimination. The preprocessing system focuses on transforming the data while retaining the resulting feature space.

### Supervised Feature Engineering

[🧠 View Supervised Feature Engineering Notebook](https://drive.google.com/file/d/1pj4NgWMw2k3qfMl8UEL-exNZBZkLKErN/view?usp=sharing)

### Unsupervised Feature Engineering

[🧠 View Unsupervised Feature Engineering Notebook](https://drive.google.com/file/d/1BeqviRUdrkt_z7N2fwcQNTW8FxDJYl4C/view?usp=sharing)

## 🔧 Processing Components

- CSV input and dataset validation
- ID-like column detection
- Missing-value detection
- Missing-value indicators
- Numerical imputation
- Categorical imputation
- Rare-category handling
- Supervised target encoding
- Unsupervised one-hot encoding
- Skewness transformation
- Feature scaling
- Leakage-safe train/test processing
- Processed-data preview
- Pipeline information report
- CSV and ZIP downloads
- Downloadable EDA HTML report

## 🏗️ Architecture

```text
                         ┌──────────────────────────┐
                         │      Streamlit App       │
                         │         app.py           │
                         └────────────┬─────────────┘
                                      │
                                  HTTP API
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │       FastAPI API        │
                         │         main.py          │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │      Processing          │
                         │       pipeline.py        │
                         └──────────────────────────┘

EDA:
EDA notebooks → FastAPI /eda → Downloadable HTML report
```

## 📂 Project Structure

```text
.
├── app.py
├── main.py
├── pipeline.py
├── operation_worker.py
├── requirements.txt
├── README.md
│
├── EDA_supervised.ipynb
├── EDA_unsupervised.ipynb
├── Feature_Engineering_supervised.ipynb
└── Feature_Engineering_unsupervised.ipynb
```

| File | Purpose |
|---|---|
| `app.py` | Streamlit frontend and workflow controls |
| `main.py` | FastAPI backend and API endpoints |
| `pipeline.py` | Core preprocessing and feature-engineering logic |
| `operation_worker.py` | Background worker for long-running/cancelable operations |
| `requirements.txt` | Python dependencies |

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run Locally

### Start FastAPI

```bash
python main.py
```

or:

```bash
uvicorn main:app --reload
```

The API should be available at:

```text
http://localhost:8000
```

### Start Streamlit

Open another terminal:

```bash
streamlit run app.py
```

## 🌐 Deployment

### Frontend

Streamlit Community Cloud:

https://automated-ml-preprocessing.streamlit.app/

### Backend

Render:

https://automated-ml-preprocessing-api.onrender.com/

The frontend sends API requests to the deployed backend for EDA generation and preprocessing.

## 📥 Outputs

Depending on the workflow, the application can provide:

```text
X_train.csv
X_test.csv
pipeline_info.txt
processed_dataset.zip
EDA_report.html
```

### Pipeline report

The pipeline report summarizes processing decisions such as:

- detected ID columns
- missing-value handling
- transformed/skewed features
- scaling
- final feature counts
- feature-selection status

Feature selection is intentionally disabled.

## 🔄 Example Workflow

### Supervised Learning

```text
Upload Dataset
      ↓
Select Supervised Learning
      ↓
Select Dataset Workflow
      ↓
Select Target
      ↓
(Optional) Generate EDA Report
      ↓
Process Dataset
      ↓
Download Processed Data
```

### Unsupervised Learning

```text
Upload Dataset
      ↓
Select Unsupervised Learning
      ↓
Select Dataset Workflow
      ↓
(Optional) Generate EDA Report
      ↓
Process Dataset
      ↓
Download Processed Data
```

## ⏳ Operation Controls

While EDA or preprocessing is running:

- other operations are locked
- learning-type controls are locked
- dataset-workflow controls are locked
- upload controls are locked
- only the Cancel button remains available

EDA and preprocessing are independent operations.

The EDA report may take longer than preprocessing because it runs multiple analyses and generates embedded visualizations.

## ⚠️ Limitations

- CSV input is currently supported.
- Large datasets may require additional CPU and memory.
- EDA report generation can take time because it performs multiple analyses and generates visualizations.
- Cancellation stops the frontend worker waiting for an API request; if the backend has already received the request, server-side computation may continue.
- Automatic model-based feature selection is intentionally not performed.
- The project currently focuses on preprocessing and feature engineering rather than automated model training.

## 🔮 Future Improvements

- Automated supervised model selection
- Cross-validation
- Hyperparameter tuning
- Model evaluation and comparison
- Automated model export
- Model deployment
- Server-side job IDs and true backend cancellation
- Support for additional file formats
- Authentication and API access control
- Automated ML experiment tracking

## 📚 Project Notebooks

### Exploratory Data Analysis

- [Supervised EDA](https://drive.google.com/file/d/1Gh45XjIZmRMz7uWzy8_q7E62CyxnYduW/view?usp=sharing)
- [Unsupervised EDA](https://drive.google.com/file/d/1aKTGNc1MWjLFWFfCirj0hwr3H_c87wNn/view?usp=sharing)

### Feature Engineering

- [Supervised Feature Engineering](https://drive.google.com/file/d/1pj4NgWMw2k3qfMl8UEL-exNZBZkLKErN/view?usp=sharing)
- [Unsupervised Feature Engineering](https://drive.google.com/file/d/1BeqviRUdrkt_z7N2fwcQNTW8FxDJYl4C/view?usp=sharing)

## 👨‍💻 Author

**Koustav Pattanayak**

GitHub: http://github.com/Koustav-2003

LinkedIn: www.linkedin.com/in/kpattanayak

## 📄 License

MIT License
