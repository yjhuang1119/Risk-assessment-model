from model_train import model_train
from model_featureImportance import model_feature_importance
from model_eveluate import prob_threshold, score

def main(train_data, valid_data, test_data, y_col_name='PHENO', model_params=None):
    """
    Main function to train, evaluate, and save results of the risk assessment model.

    Parameters:
    train_data (pandas.DataFrame): Training dataset.
    valid_data (pandas.DataFrame): Validation dataset.
    test_data (pandas.DataFrame): Test dataset.
    y_col_name (str, optional): Name of the target column. Defaults to 'PHENO'.
    model_params (dict, optional): Dictionary containing model parameters. Defaults to None.

    Returns:
    None
    """
    
    # Default parameters
    default_params = {
        'gamma': 0,
        'max_depth': 6,
        'subsample': 0.7,
        'min_child_weight': 1,
        'scale_pos_weight': 1,
        'eta': 0.3
    }

    if model_params is None:
        model_params = default_params

    # Model training
    model = model_train(train_data, valid_data, y_col_name=y_col_name, **model_params)

    # Model feature importance
    feature_importance = model_feature_importance(model, train_data, y_col_name=y_col_name)

    # Model evaluation
    threshold_final = prob_threshold(model, valid_data, y_col_name=y_col_name)
    evaluation_results = score(model, test_data, threshold_final, y_col_name=y_col_name)

    # Output result files (feature importance & evaluation)
    feature_importance.to_csv("feature_importance.csv")
    evaluation_results.to_csv("evaluation_results.csv", index=False)
