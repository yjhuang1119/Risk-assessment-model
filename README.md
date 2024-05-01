# Risk Assessment Model

This repository contains code for building and evaluating a risk assessment model using XGBoost. The model is designed to take input data, train an XGBoost model, and provide evaluation metrics such as performance index and feature importance.


## Input Data Format
Users can use any data readable into Python and convert it into a pandas DataFrame. You can refer to the example in `usage_example.py` for guidance.
The dataset should contain predictor variables and a target variable, which should be binary (0 or 1) for binary classification tasks.


## Model Parameters
The model accepts the following parameters:
- `gamma`: The minimum loss reduction required for splitting. (range: [0,∞])
- `max_depth`: The maximum depth of a tree. (range: [0,1])
- `subsample`: The subsample ratio of the training instance. (range: [0,1])
- `min_child_weight`: The minimum sum of instance weight required in a child. (range: [0,∞])
- `scale_pos_weight`: The control of the balance between positive and negative weights.
                      <br>A typical value to consider: sum(negative instances) / sum(positive instances)
- `eta`: Learning rate. (range: [0,1])

The default values for these parameters are as follows:
- `gamma`: 0
- `max_depth`: 6
- `subsample`: 1
- `min_child_weight`: 1
- `scale_pos_weight`: 1
- `eta`: 0.3


## Output Results
The model produces the following outputs:
- `evaluation_results.csv`: Includes metrics such as AUC, Accuracy, Sensitivity, Specificity, and F1-score.
- `confusion_matrix.png`: The plot provides a visual representation of model predictions versus actual values.
- `feature_importance.csv`: Importance scores for predictor variables.


## Usage

To use this risk assessment model, follow these steps:

1. Clone this repository using the following git command:

```
git clone https://github.com/yjhuang1119/Risk-assessment-model.git
```

Alternatively, download the source files from the [GitHub website](https://github.com/yjhuang1119/Risk-assessment-model).

2. Ensure you have Python installed on your machine.

3. Install the required Python libraries by running:
   ```
   pip install -r requirements.txt
   ```

4. **Running an example directly:**
   Optionally, modify the parameters in `main.py`:
   - `model_params`: Dictionary containing model parameters (optional).
   
   ```bash
   python main.py
   ```

5. **Using your own data:**
   Prepare your datasets in CSV format, and modify the following variables in `main.py`:
   - `train_data`: Path to your training dataset.
   - `valid_data`: Path to your validation dataset.
   - `test_data`: Path to your test dataset.
   
   Modify the parameters in `main.py`:
   - `y_col_name`: Name of the target column.
   - `model_params`: Dictionary containing model parameters (optional).
   
   ```bash
   python main.py
   ```


## File Summary
- `model_train.py`: Contains functions for training the XGBoost model.
- `model_featureImportance.py`: Contains functions for calculating feature importance.
- `model_evaluate.py`: Contains functions for evaluating the model performance.
- `main.py`: Entry point for users to input their data and run the model.
- `SimulatedData_example.csv`: A simulated dataset for users to run as an example.
- `Usage_example.py`: Contains example code and data for reference. Please modify the data to run with your own datasets.
- `README.md`: This file provides information about the project.
