# LEPTOLIC

Round 3 development (NHEFS, same eight-year reported-diagnosis target):
the selected approach is now [repeated nested cross-validation](round3/CV_PROTOCOL_TH.md).
start with the [Thai evidence review and plan](round3/REVIEW_TH.md),
then [the round 3 notebook](LEPTOLIC_Round3_Review.ipynb) or
[CLI instructions](round3/README.md).
The new workflow has software tests, but real-data training remains unverified
because the development environment blocks CDC downloads with proxy HTTP 403.
The original notebook and its saved outputs are preserved for comparison.

**An Experimental 8-Year Future Hypertension Prediction Model Using NHEFS Data**

LEPTOLIC is a machine learning research prototype that estimates the probability of a participant receiving their first reported physician diagnosis of hypertension within eight years, based on historical baseline health data.

The project uses data from the **National Health and Nutrition Examination Survey Epidemiologic Follow-up Study (NHEFS)** provided by the U.S. National Center for Health Statistics (NCHS), part of the CDC.

The entire workflow is implemented in a Jupyter Notebook and is designed to run in **Google Colab using CPU**, without requiring paid APIs or external machine learning services.

## Project Overview

Unlike a model that identifies whether someone currently has hypertension, LEPTOLIC focuses on a future outcome:

> Whether an individual reports receiving their first physician diagnosis of hypertension within eight years after their baseline interview in 1982–1984.

The project follows these main stages:

1. Download historical NHEFS datasets from the CDC.
2. Merge participant records across multiple follow-up waves.
3. Construct an eight-year prediction target.
4. Prepare and preprocess the dataset.
5. Train a Logistic Regression model.
6. Calibrate predicted probabilities using Platt Scaling.
7. Evaluate model performance on an independent test set.
8. Allow users to experiment with input values and obtain prediction scores.
9. Export the model, evaluation results, and processed datasets.

## Dataset

**Source:** National Health and Nutrition Examination Survey Epidemiologic Follow-up Study (NHEFS).

Official documentation:

- [CDC NHEFS Documentation](https://wwwn.cdc.gov/nchs/nhanes/nhefs/default.aspx)
- [1982 Interview](https://wwwn.cdc.gov/nchs/data/nhefs/1982intv.pdf)
- [1986 Interview](https://wwwn.cdc.gov/nchs/data/nhefs/1986intv.pdf)
- [1987 Interview](https://wwwn.cdc.gov/nchs/data/nhefs/1987intv.pdf)
- [1992 Interview](https://wwwn.cdc.gov/nchs/data/nhefs/1992intv.pdf)

### Data Collection Waves

The project combines four historical interview waves:

| Wave | Purpose |
|---|---|
| 1982–1984 | Baseline participant information |
| 1986 | First follow-up |
| 1987 | Second follow-up |
| 1992 | Final follow-up |

Participant records are matched using their corresponding identification fields across survey waves.

### Target Construction

The target represents whether a participant received their first reported physician diagnosis of hypertension within eight years after the baseline interview.

- `1`: First reported diagnosis within eight years.
- `0`: No reported diagnosis within eight years, supported by sufficient follow-up information.

Cases with insufficient follow-up, unknown responses, inconsistent diagnosis histories, or uncertain diagnosis dates are excluded rather than incorrectly labeled as negative cases.

The cohort is restricted to participants who reported no previous diagnosis at baseline (`D2 = 2`) and had baseline measured blood pressure below the project's eligibility thresholds:

- Systolic Blood Pressure (SBP) below 130 mmHg.
- Diastolic Blood Pressure (DBP) below 80 mmHg.

## Features

The current model uses five baseline features:

| Feature | Description | Unit |
|---|---|---|
| `age_years` | Participant's age | Years |
| `sex_female` | Sex encoded as 0 = Male, 1 = Female | Binary |
| `baseline_sbp` | Baseline systolic blood pressure | mmHg |
| `baseline_dbp` | Baseline diastolic blood pressure | mmHg |
| `sleep_hours` | Average sleep duration | Hours/night |

Blood pressure values are derived from available baseline measurements. Sleep duration is optional and can be imputed using the training-set median.

**Note:** The current implementation does not use the original 11-feature schema, BMI, or dietary features as model inputs.

## Machine Learning Model

LEPTOLIC uses Logistic Regression with L2 (Ridge) regularization.

### Training Pipeline

The dataset is divided into four stratified subsets:

| Dataset | Percentage | Purpose |
|---|---:|---|
| Training | 55% | Fit preprocessing and model coefficients |
| Tuning | 15% | Select the regularization strength |
| Calibration | 15% | Calibrate predicted probabilities |
| Testing | 15% | Independent performance evaluation |

The random seed is fixed at `42` to make the data split reproducible.

### Model Optimization

Five regularization strengths are evaluated:

`0.01, 0.1, 1.0, 10.0, 100.0`

The best configuration is selected using the Brier Score on the tuning set.

### Probability Calibration

Platt Scaling is applied using the separate calibration dataset to improve the reliability of predicted probabilities.

The final model is also compared against a constant baseline that predicts the event prevalence observed in the training dataset.

### Evaluation Metrics

The project evaluates the model using:

- ROC-AUC
- Brier Score
- Log Loss
- Accuracy
- Precision
- Recall
- F1 Score
- Balanced Accuracy
- Confusion Matrix
- Calibration Analysis

A 95% bootstrap confidence interval is also calculated for the test ROC-AUC.

All evaluation results are generated from the actual dataset during execution. No performance scores are assumed or hardcoded in this README.

## Getting Started

### Requirements

- Python 3
- NumPy
- Google Colab (recommended)
- Internet connection for downloading CDC datasets

The notebook uses NumPy for model training and does not require paid machine learning APIs.

### Running the Project

1. Open the notebook:

   `LEPTOLIC_NHEFS_Future_Prediction.ipynb`

2. Open it in [Google Colab](https://colab.research.google.com/).

3. Select **Runtime → Run all**.

4. Allow the notebook to download the official NHEFS datasets from the CDC.

5. Wait for the data preparation, model training, calibration, and evaluation processes to complete.

6. Review the generated evaluation report.

7. Experiment with different input values in the prediction section.

8. Download the generated output ZIP archive.

**Important:** The pipeline requires successful access to the official CDC datasets. It will stop if downloading or validating the source data fails. It does not silently substitute synthetic data.

## Making Predictions

The notebook provides two ways to experiment with the model:

1. Modify the `INPUT` dictionary directly.
2. Use the interactive input form provided through `ipywidgets` in Google Colab.

Example:

```python
INPUT = {
    'age_years': 50,
    'sex_female': 0,
    'baseline_sbp': 120,
    'baseline_dbp': 75,
    'sleep_hours': 7
}

PREVIOUSLY_DIAGNOSED = False

answer = predict_future(
    model_bundle,
    INPUT,
    previously_diagnosed=PREVIOUSLY_DIAGNOSED
)

print(answer['experimental_probability_8y'])
```

The prediction function validates input values against the training data range and the project's baseline eligibility criteria.

Missing sleep duration can be imputed, while other required features must be provided.

## Output Files

After successful execution, the project generates `LEPTOLIC_NHEFS_Future_Outputs.zip`.

The archive contains:

| File | Description |
|---|---|
| `NHEFS_Merged_Four_Waves_Raw.csv` | Merged historical participant records |
| `NHEFS_Target_Audit.csv` | Target construction and exclusion reasons |
| `NHEFS_Future_8y_Cohort.csv` | Prepared dataset with reliable labels |
| `NHEFS_Data_Audit.json` | Dataset summary and source information |
| `NHEFS_Model_Data.csv` | Model features, targets, and dataset splits |
| `NHEFS_Future_Model.json` | Trained model parameters and preprocessing configuration |
| `NHEFS_Model_Metrics.json` | Model evaluation results |
| `NHEFS_Test_Predictions.csv` | Predictions for the held-out test set |
| `NHEFS_Test_Calibration.csv` | Test-set calibration results |
| `NHEFS_Feature_Dictionary.csv` | Feature definitions and data sources |
| `NHEFS_Future_Report.md` | Generated model evaluation and limitations report |

## Limitations

LEPTOLIC is an experimental research prototype, not a clinically validated prediction system.

Important limitations include:

- **Historical population:** NHEFS participants represent a historical U.S. population, not modern Thai individuals.
- **Reported diagnosis:** The target is based on self-reported or proxy-reported physician diagnosis, not verified clinical onset.
- **Selection bias:** Participants with sufficiently reliable follow-up information may differ from those excluded from the analysis.
- **Incomplete follow-up:** Unknown, missing, and insufficient follow-up cases are excluded instead of being treated as negative outcomes.
- **Year-only diagnosis dates:** Diagnosis years overlapping the eight-year prediction boundary are excluded because the exact diagnosis date is unavailable.
- **No competing-risk adjustment:** The current implementation does not apply inverse probability of censoring weighting (IPCW) or competing-risk correction.
- **No national survey estimates:** The current model does not produce survey-weighted national risk estimates.

The predicted probability should therefore be interpreted as an experimental score conditional on the selected historical cohort, not as an individual's clinically established risk.

**This project is not intended for medical diagnosis, treatment decisions, or clinical use.**

## Disclaimer

This project is an independent research and educational prototype using publicly available NHEFS data.

The analyses and interpretations are those of the project author and do not represent the official views of the CDC or NCHS.

---

**Project:** LEPTOLIC  
**Model:** Ridge Logistic Regression + Platt Scaling  
**Dataset:** CDC NHEFS  
**Prediction Horizon:** 8 Years  
**Execution Environment:** Google Colab (CPU)
