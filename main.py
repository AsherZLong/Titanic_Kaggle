# import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

y = train_data.Survived
useful_features = ['Age', 'Pclass', 'Fare', 'SibSp', 'Parch']
X = train_data[useful_features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

my_imputer = SimpleImputer(strategy='constant', fill_value = 0)
i_train_X = pd.DataFrame(my_imputer.fit_transform(train_X))
i_val_X = pd.DataFrame(my_imputer.transform(val_X))


def get_mae(i_train_X, i_val_X, train_y, val_y):
    forest_model = RandomForestRegressor(n_estimators=10, max_depth=20,
                                         random_state=1)
    forest_model.fit(i_train_X, train_y)
    predictions = forest_model.predict(i_val_X)
    return mean_absolute_error(val_y, predictions)


my_mae = get_mae(train_X, val_X, train_y, val_y)

print(f'Using mean method:, {my_mae}')


# EDA
# training data vs survivor data shows survivros were: on average slightly younger (28.3 vs 29.7); they were higher class (avg 1.95 vs 2.31);
# had less Siblings/Spouses (0.474 vs 0.523); had more Parents/Children (0.465 vs 0.382) and paid more (48.4 vs 32.2).
# There are 177 empty age entries in train data.