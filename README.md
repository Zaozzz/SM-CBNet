# SM-CBNet: A Speech-Based Parkinson’s Disease Diagnosis Model with SMOTE–ENN and CNN+BiLSTM

## Introduction

This is a repository for reproducing the paper [SM-CBNet: A Speech-Based Parkinson’s Disease Diagnosis Model with SMOTE–ENN and CNN+BiLSTM](https://link.springer.com/chapter/10.1007/978-981-95-0030-7_4), which has been accepted as an oral presentation at the 2025 International Conference on Intelligent Computing (ICIC 2025).

## Usage
### Prepare Datasets
We use a public datasets in our model, they can be downloaded from:

[Parkinsons UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/174/parkinsons)

[Parkinson's Disease Classificatio UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification)

### 🔧 Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourname/SM-CBNet.git
cd SM-CBNet

# 2. (Recommended) Create a dedicated Conda environment
conda create -n smcbnet python=3.10 -y
conda activate smcbnet

# 3. Install dependencies
pip install -r requirements.txt
````
---

### 📁 Project Structure

```
SM-CBNet
├── data/                         # Place your datasets here
│   └── parkinsons.csv            # Example merged CSV
├── dataload.py                   # Data loading + SMOTE–ENN oversampling
├── model.py                      # CNN+BiLSTM architecture
├── main.py                       # Training & evaluation entry point
├── requirements.txt              # Dependency list
└── README.md
```

---

### 🚀 Quick Start

```bash
python main.py \
  --data data/parkinsons.csv \
  --target status \
  --epochs 10 \
  --batch 32
```

| Argument          | Description                    | Default |
| ----------------- | ------------------------------ | ------- |
| `--no_oversample` | Disable SMOTE–ENN oversampling | Off     |
| `--epochs`        | Number of training epochs      | 10      |
| `--batch`         | Batch size                     | 32      |

After training, the script automatically prints a **Confusion Matrix** and **Classification Report**.

---

### 🔍 Inference

```python
import pandas as pd
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("saved_model/smcbnet.h5")

# Read and preprocess new samples
df_new = pd.read_csv("data/new_cases.csv")
X = df_new.values.reshape(df_new.shape[0], -1, 1)  # (samples, timesteps, 1)

# Predict
probs = model.predict(X)
preds = (probs > 0.5).astype(int).flatten()
print(preds)
```

---




## Citation
If you think that our work is useful to your research, please cite using this BibTeX:
```bibtex
@InProceedings{10.1007/978-981-95-0030-7_4,
author="Wang Xu, Pan Weichao, Liu Ruida, Tian Zhen, Jin Keyan",
title="SM-CBNet: A Speech-Based Parkinson's Disease Diagnosis Model with SMOTE--ENN and CNN{\thinspace}+{\thinspace}BiLSTM Integration",
booktitle="Advanced Intelligent Computing Technology and Applications",
year="2025",
publisher="Springer Nature Singapore",
address="Singapore",
pages="40--51",
abstract="Parkinson's disease (PD) is the second most prevalent neurodegenerative disorder worldwide. Speech-based diagnostic approaches for PD have attracted increasing attention, with deep learning models demonstrating promising performance. In this paper, we propose a speech-based diagnostic model for PD, aiming to enhance the diagnostic accuracy using deep learning techniques. We adopt the SMOTE--ENN oversampling method to solve the data imbalance problem, and develop a hybrid model that integrates a Convolutional Neural Network (CNN) and Bi-directional Long and Short-Term Memory network (BiLSTM) to efficiently extract the speech features and capture temporal dependencies. Experimental results show that the proposed model achieves an accuracy of 95{\%} on public datasets and outperforms traditional machine learning and other deep learning models in several evaluation metrics, validating the effectiveness of our network in Parkinson's disease diagnosis. These results validate the effectiveness of our approach and highlight its potential for high-precision early screening of PD, offering reliable technical support for clinical applications.",
isbn="978-981-95-0030-7"}
```
If you have questions about this repo, please submit an issue or contact [Xu Wang](mailto:zaowxx@163.com).
