# CS5100 Final Project: Student Success Prediction

**Foundations of Artificial Intelligence, Fall 2025**

## Project Overview

This project implements a complete machine learning pipeline to predict student academic risk using the UCI Student Performance dataset. The system identifies students at risk of failing (final grade < 10) to enable early intervention.

### Key Achievements
- **Phase 1 Complete**: Baseline models (Gradient Boosting + from-scratch Random Forest)
- **Phase 2 Complete**: full dataset, feature selection, stacking ensemble
- **Best Model**: F1 = 0.5085 using Gradient Boosting with RFE feature selection
- **All Tests Passing**: 18/18 Phase 1 tests validated

## Repository Structure

```
CS5100-Final-Project-main/
├── student_project/
│   └── student_project.py          # Phase 1 implementation
├── datasets/
│   ├── student-mat.csv             # Full UCI dataset (395 samples)
│   ├── student-mat-mini.csv        # Mini dataset (39 samples)
│   └── student.txt                 # Dataset documentation
├── tests/
│   ├── conftest.py
│   └── test_phase_1.py             # Phase 1 test suite
├── Report.pdf
├── phase2_implementation.py        # Phase 2 advanced techniques
└── README.md                       # This file
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install pandas numpy scikit-learn pytest

```

### Running Phase 1

```bash
# Run Phase 1
python student_project.py
```

### Running Phase 2 Implementation

```bash
# Execute full Phase 2 pipeline
python phase2_implementation.py
```

## Phase 1: Baseline Implementation

### Components Implemented

#### 1. Data Loading (`load_data`)

#### 2. Data Preprocessing (`preprocess_data`)

#### 3. Gradient Boosting Pipeline (`train_gb_pipeline`)

#### 4. Random Forest (From Scratch)

### Key Formulas

**Gini Impurity**: 
```
G = 1 - Σ(p_i²)
```
where p_i is the proportion of samples in class i

**Min-Max Scaling**:
```
x_scaled = (x - x_min) / (x_max - x_min)
```

## Phase 2: Advanced Techniques

#### 1. Full Dataset Usage
- Transitioned from mini (39) to full (395) dataset
- 10x increase in training samples
- **Impact**: GB F1 improved from 0.2857 to 0.4815 (+68%)

#### 2. Feature Selection 
**Mutual Information**:
- SelectKBest with mutual_info_classif
- Top 20 features selected
- Key features: Medu, failures, freetime, goout, Dalc
- **Performance**: F1 = 0.4364, Accuracy = 0.6869, AUC = 0.6630

**Recursive Feature Elimination (RFE)** **Best**:
- Iterative feature removal with GB base estimator
- Top 15 features selected
- Key features: age, Medu, Fedu, studytime, failures
- **Performance**: F1 = 0.5085, Accuracy = 0.7071, AUC = 0.6699
- **Improvement**: +0.0270 over baseline GB

#### 3. Stacking Ensemble
**Architecture**:
- Level 0 (Base Models):
  - GB1: GradientBoosting (100 est, depth 3, lr 0.1)
  - GB2: GradientBoosting (150 est, depth 4, lr 0.05)
  - LR: LogisticRegression
- Level 1 (Meta-Model):
  - LogisticRegression with 5-fold CV

**Results**:
- Standard: F1 = 0.3111, AUC = 0.6791
- With Feature Selection: F1 = 0.3673, AUC = 0.6882
- **Analysis**: Underperformed baseline, demonstrating that complexity doesn't guarantee improvement on small datasets

## Performance Summary

### Full Dataset Results (395 samples, 75/25 split)

| Model | F1 Score | Accuracy | ROC-AUC |
|-------|----------|----------|---------|
| Gradient Boosting (Baseline) | 0.4815 | 0.7172 | 0.6708 |
| Random Forest (Scratch) | 0.3333 | 0.6768 | N/A |
| GB + Mutual Information | 0.4364 | 0.6869 | 0.6630 |
| **GB + RFE (Best)** | **0.5085** | **0.7071** | **0.6699** |
| Stacking Ensemble | 0.3111 | 0.6869 | 0.6791 |
| Stacking + Feature Selection | 0.3673 | 0.6869 | 0.6882 |

**Best Model**: Gradient Boosting with RFE feature selection achieves F1 = 0.5085

## Key Insights

1. **Dataset Size Matters**: Moving from mini to full dataset improved F1 by 68%
2. **Feature Selection Helps**: RFE reduced features from 39 to 15 while improving F1
3. **Ensemble Trade-offs**: Stacking underperformed, showing complexity isn't always better
4. **Custom Implementation Viable**: From-scratch Random Forest achieved competitive results
5. **Important Features**: Parental education, study time, and past failures are key predictors

## Technical Details

### Data Preprocessing Pipeline
1. Create binary target: `at_risk = (G3 < 10).astype(int)`
2. Remove grade columns (G1, G2, G3) to prevent leakage
3. One-hot encode categorical variables (school, sex, address, etc.)
4. Scale numerical features to [0, 1] using min-max normalization
5. Impute missing values (median for numeric, mode for categorical)

### Model Training
- Train/test split: 75/25 with stratification
- Random state: 42 for reproducibility
- Cross-validation: 5-fold for stacking meta-model
- Evaluation metrics: F1 score (primary), accuracy, ROC-AUC

### Metric Gates (Phase 1)
- **Gradient Boosting**: F1 ≥ 0.40 OR F1 ≥ (dummy + 0.10) OR AUC ≥ 0.50 
- **Random Forest**: F1 ≥ 0.30 OR F1 ≥ (GB − 0.15) 

## Dependencies

### Required
- Python 3.8+
- pandas >= 1.5
- numpy >= 1.23
- scikit-learn >= 1.2

## Citation

Dataset: UCI Machine Learning Repository
- Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance.
