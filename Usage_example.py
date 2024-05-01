from sklearn.model_selection import train_test_split
import pandas as pd

from model_train import model_train
from model_featureImportance import model_feature_importance
from model_eveluate import prob_threshold, score
from main import main

#### read example data ####
data = pd.read_csv('./example_data/SimulatedData_example.csv',index_col=0)

# Define a list of categorical columns 
cate_cols = ['SEX', 'ECG_summary', 'liver_tumor', 'liver_cyst', 'liver_cal', 'liver_hemangioma', 
             'liver_fibrosis', 'liver_chronic', 'liver_fatty', 'GB_polyp', 'GB_stone', 
             'CBD_dilatation', 'CBD_stone', 'pancreas_cyst', 'pancreas_tumor', 'pancreas_cal', 
             'pancreas_stone', 'pancreas_dilatation', 'pancreas_pancreatitis', 'spleen_tumor', 
             'spleen_cyst', 'spleen_cal', 'spleen_accessory', 'spleen_splenomegaly', 'kidney_cyst', 
             'kidney_stone', 'kidney_cal', 'kidney_hydronephrosis', 'kidney_polycystic', 
             'kidney_tumor', 'MEASURE_BONE_POSE']

# Convert specified columns to categorical data type #
data[cate_cols] = data[cate_cols].astype('category')

# split data to train, valid, test #
trainval, test_data = train_test_split(
data,test_size=0.2,stratify=data.PHENO, random_state=1)

train_data, valid_data = train_test_split(
trainval,test_size=0.2,stratify=trainval.PHENO, random_state=1)


#### Train the model and return results ####

## default parameter ##
main(train_data, valid_data, test_data)

'''
## specific paramter ##

# Name of the target column
y_col_name = 'PHENO'  

# Define specific model parameters
model_params = {
    'max_depth': 3,
    'subsample': 0.8
}

main(train_data, valid_data, test_data, y_col_name, model_params)
'''