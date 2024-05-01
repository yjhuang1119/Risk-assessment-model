import pandas as pd
import numpy as np

#output feature importance
def model_feature_importance(model, train_data,y_col_name='PHENO'):
    Feature_df = []
    f = ['weight','gain','cover']
    for i in f:
        Feature_df.append(model.get_score(importance_type=i))
    
    Feature_df=pd.DataFrame(Feature_df,index=f).T

    var_notuse=set(train_data.drop([y_col_name],axis=1).columns) - set(Feature_df.index)
    var_notuse=pd.DataFrame({'weight':np.repeat(0,len(var_notuse)),
                'gain':np.repeat(0,len(var_notuse)),
                'cover':np.repeat(0,len(var_notuse))},index=list(var_notuse))

    Feature_df=pd.concat([Feature_df,var_notuse],axis=0)

    return Feature_df