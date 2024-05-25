# import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

y = train_data.Survived
useful_features = ['Age', 'Pclass', 'Fare', 'SibSp', 'Parch', 'Embarked']
X = train_data[useful_features]
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)

'''my_imputer = SimpleImputer()
i_train_X = pd.DataFrame(my_imputer.fit_transform(train_X))
i_val_X = pd.DataFrame(my_imputer.transform(val_X))'''


def get_mae(train_X, val_X, train_y, val_y):
    forest_model = RandomForestRegressor(random_state=1)
    forest_model.fit(train_X, train_y)
    predictions = forest_model.predict(val_X)
    return mean_absolute_error(val_y, predictions)


# my_mae = get_mae(i_train_X, i_val_X, train_y, val_y)

# EDA
# training data vs survivor data shows survivros were: on average slightly younger (28.3 vs 29.7); they were higher class (avg 1.95 vs 2.31);
# had less Siblings/Spouses (0.474 vs 0.523); had more Parents/Children (0.465 vs 0.382) and paid more (48.4 vs 32.2).

# There are 177 empty age entries in train data.
# Found mode imputation reduced MAE the most but MAE may not be good indicator.

categorical_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 and 
                    X_train[cname].dtype == "object"]
numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

num_transformer = SimpleImputer(strategy='constant')

category_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_cols),
        ('cat', category_transformer, categorical_cols)
    ])

model = RandomForestRegressor(n_estimators=100, random_state=0)
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_val)

# Evaluate the model
score = mean_absolute_error(y_val, preds)
print('MAE:', score)

