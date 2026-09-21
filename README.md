# MAGIC Gamma Telescope Classification using Naive Bayes

A Machine Learning workflow that classifies Cherenkov gamma telescope signals (`g`) versus background hadron noise (`h`) using probabilistic modelling with Gaussian Naive Bayes and automated class balancing.

---

## Technical Overview

- **Algorithm**: Gaussian Naive Bayes (`sklearn.naive_bayes.GaussianNB`)
- **Dataset**: MAGIC Gamma Telescope Dataset (`19,020` samples, `10` continuous features)
- **Preprocessing**: Standard Scaling (`StandardScaler`) & Oversampling (`RandomOverSampler`)
- **Data Split**: 60% Train / 20% Validation / 20% Test

---

## Features

| Feature Column | Description |
| :--- | :--- |
| `fLength` | Major axis of ellipse [mm] |
| `fWidth` | Minor axis of ellipse [mm] |
| `fSize` | 10-log of sum of content of all pixels |
| `fConc` | Ratio of sum of two highest pixels to fSize |
| `fConc1` | Ratio of highest pixel to fSize |
| `fAsym` | Distance from highest pixel to center |
| `fM3Long` | 3rd root of 3rd moment along major axis |
| `fM3Trans` | 3rd root of 3rd moment along minor axis |
| `fAlpha` | Angle of major axis with vector to origin |
| `fDist` | Distance from origin to center of ellipse |
| `class` | Target Variable: Gamma (`g` $\rightarrow$ `1`), Hadron (`h` $\rightarrow$ `0`) |

---

## Quickstart & Implementation

### Prerequisites

```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn
