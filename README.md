# Risk Assessment Model

## Introduction
This repository contains code for building and evaluating a risk assessment model using XGBoost. The model takes input data from users, trains an XGBoost model, and provides evaluation metrics such as performance index and feature importance.

## Input Data Format
Users can input their data sets in CSV format. The data set should contain predictor variables and a target variable. The target variable should be binary (0 or 1) for binary classification tasks.

## Model Parameters
The model accepts the following parameters:
- `gamma`: Gamma parameter for regularization.
- `max_depth`: Maximum depth of a tree.
- `subsample`: Subsample ratio of the training instance.
- `min_child_weight`: Minimum sum of instance weight (hessian) needed in a child.
- `scale_pos_weight`: Control the balance of positive and negative weights.
- `eta`: Learning rate.

The default values for these parameters are as follows:
- `gamma`: 0
- `max_depth`: 6
- `subsample`: 0.7
- `min_child_weight`: 1
- `scale_pos_weight`: 1
- `eta`: 0.3

## Output Results
The model produces the following outputs:
- Performance index: Includes metrics such as AUC, Accuracy, Sensitivity, Specificity, and F1-score.
- Comfusion matrix: The plot provides a visual representation of model predictions versus actual values.
- Feature Importance: Importance scores for predictor variables.


## File Summary
- `model_train.py`: Contains functions for training the XGBoost model.
- `model_featureImportance.py`: Contains functions for calculating feature importance.
- `model_evaluate.py`: Contains functions for evaluating the model performance.
- `main.py`: Entry point for users to input their data and run the model.
- `SimulatedData_example.csv`: A simulated dataset for users to run as an example.
- `Usage_example`: An example illustrating how to use the code.
- `README.md`: This file provides information about the project.

