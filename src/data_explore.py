import pandas as pd
from sklearn.tree import DecisionTreeRegressor

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

model = DecisionTreeRegressor(random_state=1)
model.fit(X, y)

# making predictions on the first few rows
print("Making predictions for the following 5 houses")
print(X.head())
print("The predictions are:")
print(model.predict(X.head()))