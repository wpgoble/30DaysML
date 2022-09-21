import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

file_src = '../data/melb_data.csv'
df = pd.read_csv(file_src)

# Print a summary of the data from our dataset
print(df.describe())
print(df.columns)
print("==="*38)

# for now drop missing values
df = df.dropna(axis=0)      # axis 0 is drop the row with the missing data

# select our prediction target
y = df.Price

# select the feautres we want to use to predict the home price
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = df[features]

print(X.describe())
print(X.head())

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

model = DecisionTreeRegressor(random_state=1)
model.fit(train_X, train_y)

# making predictions on the first few rows
# print("Making predictions for the following 5 houses")
# print(X.head())
# print("The predictions are:")
# print(model.predict(X.head()))

val_predictions = model.predict(val_X)
print(f'MAE: {mean_absolute_error(val_y, val_predictions)}')