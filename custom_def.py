def catconsep(df):
    cat = list(df.columns[df.dtypes=='object'])
    con = list(df.columns[df.dtypes!='object'])
    return cat, con

def replacer(df):
    cat, con = catconsep(df)
    for i in df.columns:
        if i in cat:
            md = df[i].mode()[0]
            df[i] = df[i].fillna(md)
        else:
            mn = df[i].mean()
            df[i] = df[i].fillna(mn)
            
def evaluate_model(xtrain,ytrain,xtest,ytest,model):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    # Predict train and test values
    ypred_tr = model.predict(xtrain)
    ypred_ts = model.predict(xtest)
    
    # Evaluate for training
    tr_mse = mean_squared_error(ytrain,ypred_tr)
    tr_rmse = tr_mse**(1/2)
    tr_mae = mean_absolute_error(ytrain,ypred_tr)
    tr_r2 = r2_score(ytrain, ypred_tr)
    
    # Evaluate for testing
    ts_mse = mean_squared_error(ytest,ypred_ts)
    ts_rmse = ts_mse**(1/2)
    ts_mae = mean_absolute_error(ytest,ypred_ts)
    ts_r2 = r2_score(ytest, ypred_ts)
    
    # Print the results
    print('Training Results:')
    print(f'MSE  : {tr_mse:.2f}')
    print(f'RMSE : {tr_rmse:.2f}')
    print(f'MAE  : {tr_mae:.2f}')
    print(f'R2   : {tr_r2:.4f}')
    
    print('\n=======================\n')
    print('Testing Results:')
    print(f'MSE  : {ts_mse:.2f}')
    print(f'RMSE : {ts_rmse:.2f}')
    print(f'MAE  : {ts_mae:.2f}')
    print(f'R2   : {ts_r2:.4f}')

def r2_adj(xtrain,ytrain,model):
    """
    Only to be calculated for training Data
    """
    from sklearn.metrics import r2_score
    ypred = model.predict(xtrain)
    r = r2_score(ytrain,ypred)
    N = xtrain.shape[0]
    p = xtrain.shape[1]
    num = (1-r)*(N-1)
    den = (N-p-1)
    r_adj = 1 - (num/den)
    return r_adj

def univariate_plot(df):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    cat, con = catconsep(df)
    
    #Categorical features count plot
    print('Countplot for Categorical features')
    for i in cat:
        df[i].value_counts().plot(kind='bar', title = f'Countplot for {i}')
        plt.show()
    
    print('\n=====================================================\n')
    
    #Continuous features histogram
    for i in con:
        sns.histplot(data=df, x=i,kde=True)
        plt.title(f'Histogram for {i}')
        plt.show()
        
def ANOVA(df,target,cat):
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    frm = f'{target}~{cat}'
    model = ols(frm,df).fit()
    a = anova_lm(model)
    p_val = a.iloc[0,-1]
    return p_val   