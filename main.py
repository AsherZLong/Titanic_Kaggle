# import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

y = train_data.Survived
useful_features = ['Age', 'Pclass', 'Fare', 'SibSp', 'Parch']
X = train_data[useful_features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)


def get_mae(test_val, train_X, val_X, train_y, val_y):
    forest_model = RandomForestRegressor(n_estimators=test_val, random_state=1)
    forest_model.fit(train_X, train_y)
    predictions = forest_model.predict(val_X)
    return mean_absolute_error(val_y, predictions)


for estimators in [10, 50, 100, 200, 500, 1000]:
    my_mae = get_mae(estimators, train_X, val_X, train_y, val_y)
    print('for n_estiamtors: %d \t\t MAE: %d ' % (estimators, my_mae))


# EDA
# training data vs survivor data shows survivros were: on average slightly younger (28.3 vs 29.7); they were higher class (avg 1.95 vs 2.31);
# had less Siblings/Spouses (0.474 vs 0.523); had more Parents/Children (0.465 vs 0.382) and paid more (48.4 vs 32.2).
# There are 177 empty age entries in train data.

'''y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch", "Age"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
# for i in range(10):
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)
print(mean_absolute_error(y,predictions))

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")'''
