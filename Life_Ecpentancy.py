import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.width',500)

import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,median_absolute_error
from sklearn.ensemble import VotingRegressor,RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

train = pd.read_csv("kaggle_yarisma_new/WHO Life Expectancy Prediction/training data.csv")
test = pd.read_csv("kaggle_yarisma_new/WHO Life Expectancy Prediction/testing data.csv")
row_id = test["Row_id"]
test.drop("Row_id",axis=1,inplace=True)

train['Life expectancy '] = train['Life expectancy '].fillna(train['Life expectancy '].median())


df = pd.concat([train,test],axis=0,ignore_index=True)

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

def grab_cols_names(dataframe,cat_th=10,car_th=20):
    """

    Parameters
    ----------
    dataframe
    cat_th
    car_th

    Returns
    -------
    cat_cols,num_cols,cat_but_car
    """
    # categorical columns

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes not in ["int64","float64"]]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64","float64"]
                and dataframe[col].nunique()<cat_th]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtypes not in ["int64","float64"]
                   and dataframe[col].nunique()>car_th]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]


    # numerical columns

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64","float64"]]

    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols,num_cols,cat_but_car


cat_cols,num_cols,cat_but_car = grab_cols_names(df)

num_cols = [col for col in num_cols if col not in "'Life expectancy "]

# numerical columns analysis


def num_cols_analysis(dataframe,num_cols,plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[num_cols].describe(quantiles).T)

    if plot:
        dataframe[num_cols].hist(bins=50)
        plt.xlabel(num_cols)
        plt.title(num_cols)
        plt.show(block=True)
    print("###########################################")

for col in num_cols:
    num_cols_analysis(df,col,plot=False)

# categorical  columns analysis

def categorical_columns_analysis(dataframe, cat_cols,plot=False):
    print(pd.DataFrame({cat_cols: dataframe[cat_cols].value_counts(),
                        "Ratio": 100 * dataframe[cat_cols].value_counts() / len(dataframe)}), end="\n\n\n")
    if plot:
        sns.countplot(x=cat_cols,data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    categorical_columns_analysis(df,col,plot=True)



# Base Model
def base_model(train_df,test_df,df):
    train_df.drop("Country",axis=1,inplace=True)
    test_df.drop("Country",axis=1,inplace=True)

    label_encoder = LabelEncoder()
    train_df['Status'] = label_encoder.fit_transform(train_df['Status'])
    test_df['Status'] = label_encoder.fit_transform(test_df['Status'])

    test_df.drop("Life expectancy ",axis=1,inplace=True)

    X= train_df.drop("Life expectancy ",axis=1)
    y=train_df["Life expectancy "]

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)


    lgbm = LGBMRegressor()

    lgbm.fit(X_train,y_train)

    y_pred_base_model = lgbm.predict(X_test)

    return mean_absolute_error(y_test,y_pred_base_model)

train_df = df[df["Life expectancy "].notnull()]
test_df = df[df["Life expectancy "].isnull()]

base_model_mae=base_model(train_df,test_df,df)


# fillna()

100*df.isnull().sum() / len(df)

cat_cols,num_cols,cat_but_car = grab_cols_names(df)

num_cols = [col for col in num_cols if col not in "'Life expectancy "]

def data_fillna(dataframe,col_name,cat_col):
    dataframe[col_name] = df[col_name].fillna(df.groupby(cat_col)[col_name].transform("mean"))


for col in num_cols:
    data_fillna(df,col,"Status")


# Outliers

def outliers_threshold(dataframe,num_cols):
    quratile3 = dataframe[num_cols].quantile(0.95)
    quratile1 = dataframe[num_cols].quantile(0.05)
    interquantile_range = quratile3 - quratile1
    up_limit = quratile3 + 1.5*interquantile_range
    low_limit = quratile1 - 1.5*interquantile_range
    return up_limit,low_limit


def check_outlier(dataframe,num_cols):
    up,low = outliers_threshold(dataframe,num_cols)

    if dataframe[(dataframe[num_cols]>up) | (dataframe[num_cols]<low)].any(axis=None):
        return col,True
    else:
        return col,False


for col in num_cols:
    print(check_outlier(df,col))

def replace_outliers(dataframe,num_cols):
    up,low = outliers_threshold(dataframe,num_cols)
    dataframe.loc[(dataframe[num_cols]>up),num_cols] = up
    dataframe.loc[(dataframe[num_cols]<low),num_cols] = low


for col in num_cols:
    replace_outliers(df,col)

# Feature Engineering

## infant deaths
df.loc[df["infant deaths"] <= 5, "NEW_INFANT_DEATHS"] = "little_dead"
df.loc[(df["infant deaths"] > 5) & (df["infant deaths"] <= 29), "NEW_INFANT_DEATHS"] = "middle_dead"
df.loc[(df["infant deaths"] > 29) & (df["infant deaths"] <= 58), "NEW_INFANT_DEATHS"] = "a_lot_dead"
df.loc[df["infant deaths"] > 58, "NEW_INFANT_DEATHS"] = "very_a_lot_dead"


## BMI

df["NEW_BMI"] = pd.cut(x=df[" BMI "],bins=[0,18.5,24.9,29.9,100],labels=["Underweight_country", "Healthy_country", "Overweight_country", "Obese_country"])

## Hepatitis B


df["NEW_Hepatitis_B"] = pd.cut(x=df["Hepatitis B"],bins=[0,80,95,100],labels=["Low_Coverage_Hepatitis","Medium_Coverage_Hepatitis","High_Coverage_Hepatitis"])

## Polio
df["POLIO_COVERAGE_CATEGORY"] = pd.cut(df["Polio"],
                                       bins=[0, 80, 95, 100],
                                       labels=["Polio_Low_Coverage", "Polio_Medium_Coverage", "Polio_High_Coverage"])





## Adult Mortality


df['Mortality_Category'] = pd.cut(df['Adult Mortality'],
                                   bins=[0, 100, 200, 300, 400, 723],
                                   labels=['Adult_Mortality_Low', 'Adult_Mortality_Moderate', 'Adult_Mortality_High', 'Adult_Mortality_Very High', 'Adult_Mortality_Extreme'])


## Income composition of resources


df['Income_Composition_Category'] = pd.cut(df['Income composition of resources'],
                                           bins=[0, 0.5, 0.7, 0.8, 0.9, 1],
                                           labels=['Income_Composition_Category_Very_Low',
                                                   'Income_Composition_Category_Low',
                                                   'Income_Composition_Category_Moderate',
                                                   'Income_Composition_Category_High',
                                                   'Income_Composition_Category_Very High'],
                                           include_lowest=True)


## Health_Expenditure_to_GDP_Ratio
df['Health_Expenditure_to_GDP_Ratio'] = df['Total expenditure'] / df['GDP']

## Alcohol_to_BMI_Ratio
df['Alcohol_to_BMI_Ratio'] = df['Alcohol'] / df[' BMI ']

## Total_Thinness_Prevalence
df['Total_Thinness_Prevalence'] = df[' thinness  1-19 years'] + df[' thinness 5-9 years']




# Encoding

cat_cols,num_cols,cat_but_car = grab_cols_names(df)
num_cols = [col for col in num_cols if col not in "'Life expectancy "]

df.drop("Country",axis=1,inplace=True)

def label_encoder(dataframe,binary_cols):
    label_encoder = LabelEncoder()
    dataframe[binary_cols] = label_encoder.fit_transform(dataframe[binary_cols])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes not in ["int64","float64"] and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

def one_hot_encoder(dataframe,categorical_cols,drop_first=True):
    dataframe = pd.get_dummies(dataframe,columns=categorical_cols,drop_first=drop_first)
    return  dataframe

cat_cols = [col for col in cat_cols if col not in binary_cols ]

df = one_hot_encoder(df,cat_cols,drop_first=True)


# LightGBM Regressor


train_df = df[df["Life expectancy "].notnull()]
test_df = df[df["Life expectancy "].isnull()]

test_df.drop("Life expectancy ", axis=1, inplace=True)

X = train_df.drop("Life expectancy ", axis=1)
y = train_df["Life expectancy "]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


def objective_lgb(trial):
    """Define the objective function"""

    params = {
        'objective': trial.suggest_categorical('objective', ['root_mean_squared_error']),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 8, 1024),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 700, 1600),
        'min_child_weight': trial.suggest_int('min_child_weight', 10, 25),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 1.0),
        "random_state": trial.suggest_categorical('random_state', [42]),
        "extra_trees": trial.suggest_categorical('extra_trees', [True]),

    }

    model_lgb = LGBMRegressor(**params)
    model_lgb.fit(X_train, y_train)
    y_pred = model_lgb.predict(X_test)
    return mean_squared_error(y_test, y_pred, squared=False)

study_lgb = optuna.create_study(direction='minimize')
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_lgb.optimize(objective_lgb, n_trials=50,show_progress_bar=True)

study_lgb.best_params

lgbm_final = LGBMRegressor(**study_lgb.best_params)

lgbm_final.fit(X_train,y_train)

y_pred_lgbm = lgbm_final.predict(X_test)

median_absolute_error(y_test,y_pred_lgbm)

# CatBoostRegressor Regressor


def objective_cat(trial):
    """Define the objective function"""

    params = {
        'objective': trial.suggest_categorical('objective', ['RMSE']),
        'logging_level': trial.suggest_categorical('logging_level', ['Silent']),
        "random_seed": trial.suggest_categorical('random_seed', [42]),
        "iterations": trial.suggest_int("iterations", 500, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
        "depth": trial.suggest_int("depth", 5, 8),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 0.5),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 20),
        'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.01, 1),
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 10, 30),
        'reg_lambda': trial.suggest_uniform('reg_lambda', 50, 100),

    }

    model_cat = CatBoostRegressor(**params)
    model_cat.fit(X_train, y_train)
    y_pred = model_cat.predict(X_test)
    return mean_squared_error(y_test, y_pred, squared=False)

study_cat = optuna.create_study(direction='minimize')
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_cat.optimize(objective_cat, n_trials=50,show_progress_bar=True)


study_cat.best_params

catboost_final = CatBoostRegressor(**study_cat.best_params)

catboost_final.fit(X_train,y_train)

y_pred_catboost = catboost_final.predict(X_test)

median_absolute_error(y_test,y_pred_catboost)

# XGBoost Regressor

def objective_xg(trial):
    """Define the objective function"""

    params = {
        'booster': trial.suggest_categorical('booster', ['gbtree']),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'max_leaves': trial.suggest_int('max_leaves', 8, 1024),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 400, 1500),
        'min_child_weight': trial.suggest_int('min_child_weight', 10, 20),
        'subsample': trial.suggest_float('subsample', 0.3, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.01, 0.5),
        "random_state" : trial.suggest_categorical('random_state', [42]),
        'objective': trial.suggest_categorical('objective', ['reg:squarederror']),
        "n_jobs" : trial.suggest_categorical('n_jobs', [-1]),
    }

    model_xgb = XGBRegressor(**params)
    model_xgb.fit(X_train, y_train)
    y_pred = model_xgb.predict(X_test)
    return mean_squared_error(y_test,y_pred, squared=False)

study_xgb = optuna.create_study(direction='minimize')
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_xgb.optimize(objective_xg, n_trials=50,show_progress_bar=True)


study_xgb.best_params

xgb_final = XGBRegressor(**study_xgb.best_params)

xgb_final.fit(X_train,y_train)

y_pred_xgb = xgb_final.predict(X_test)


median_absolute_error(y_test,y_pred_xgb)


# Voting Regressor

voting_regressor = VotingRegressor(estimators=[
    ('catboost', catboost_final),
    ('xgb', xgb_final),
    ('lgbm', lgbm_final)
])


voting_regressor.fit(X_train,y_train)

y_pred_voting=voting_regressor.predict(X_test)

median_absolute_error(y_test,y_pred_voting)




                                # Test Data prediction


y_test_data = catboost_final.predict(test_df)

df = pd.DataFrame({"Row_id":row_id,
                   "Life expectancy":y_test_data})

df.to_csv("kaggle_yarisma_new/WHO Life Expectancy Prediction/Catboost.csv",index=False)

######################################################################


y_test_data_xgb = xgb_final.predict(test_df)
df2 = pd.DataFrame({"Row_id":row_id,
                   "Life expectancy":y_test_data_xgb})

df2.to_csv("kaggle_yarisma_new/WHO Life Expectancy Prediction/XGBRegressor.csv",index=False)

######################################################################


y_test_data_lgbm = lgbm_final.predict(test_df)
df3 = pd.DataFrame({"Row_id":row_id,
                   "Life expectancy":y_test_data_lgbm})

df3.to_csv("kaggle_yarisma_new/WHO Life Expectancy Prediction/LightGBM.csv",index=False)

######################################################################

y_test_voting = voting_regressor.predict(test_df)


df4 = pd.DataFrame({"Row_id":row_id,
                   "Life expectancy":y_test_voting})

df4.to_csv("kaggle_yarisma_new/WHO Life Expectancy Prediction/Voting.csv",index=False)

