# Importing necessary libraries
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer, roc_curve, recall_score, accuracy_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd

# Function to determine the best threshold using validation data (Yoden index)
def prob_threshold(model, valid_data, y_col_name='PHENO'):
    """
    Determine the best threshold using validation data (Yoden index).

    Parameters:
    model (xgboost.core.Booster): Trained xgboost model.
    valid_data (pandas.DataFrame): Validation dataset.
    y_col_name (str, optional): Name of the target column. Defaults to 'PHENO'.

    Returns:
    float: Best threshold.
    """
    validdata_x = valid_data.drop([y_col_name], axis=1)
    validdata_y = valid_data[y_col_name]

    dvalid = xgb.DMatrix(validdata_x, label=validdata_y, enable_categorical=True)
    
    allthreshold = np.arange(0, 1.0, 0.01)
    threshold = []
    sensitivity = []
    specificity = []
    bind = []
    for p in allthreshold:
        threshold.append(p)
        y_pred = (model.predict(dvalid) >= p).astype(int)
        tn, fp, fn, tp = confusion_matrix(validdata_y,  y_pred).ravel()
        sensitivity.append(recall_score(validdata_y, y_pred))
        specificity.append((tn / (tn + fp)))
        bind.append(recall_score(validdata_y, y_pred) + (tn / (tn + fp)))
    return threshold[np.argmax(bind)]

# Function to evaluate model performance using the best threshold and plot confusion matrix
def score(model, test_data, threshold_final, y_col_name='PHENO'):
    """
    Evaluate model performance using the best threshold and plot confusion matrix.

    Parameters:
    model (xgboost.core.Booster): Trained xgboost model.
    test_data (pandas.DataFrame): Test dataset.
    threshold_final (float): Best threshold.
    y_col_name (str, optional): Name of the target column. Defaults to 'PHENO'.

    Returns:
    pandas.DataFrame: Evaluation results including AUC, accuracy, sensitivity, specificity, and F1 score.
    """
    # Calculate AUC
    testdata_x = test_data.drop([y_col_name], axis=1)
    testdata_y = test_data[y_col_name]

    dtest = xgb.DMatrix(testdata_x, label=testdata_y, enable_categorical=True)
    
    pred = model.predict(dtest)
    auc = roc_auc_score(testdata_y, pred)

    # Apply threshold and calculate evaluation metrics
    y_pred = (model.predict(dtest) >= threshold_final).astype(int)
    accuracy = accuracy_score(testdata_y, y_pred)
    sensitivity = recall_score(testdata_y, y_pred)
    tn, fp, fn, tp = confusion_matrix(testdata_y, y_pred).ravel()
    specificity = tn / (tn + fp)
    F1 = f1_score(testdata_y, y_pred)

    # Plot confusion matrix
    cm = confusion_matrix(testdata_y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    plt.show()

    # Create evaluation results DataFrame
    evaluation_results = pd.DataFrame({
        "AUC": [auc],
        "Accuracy": [accuracy],
        "Sensitivity": [sensitivity],
        "Specificity": [specificity],
        "F1": [F1]
    })
    print(evaluation_results)

    return evaluation_results
