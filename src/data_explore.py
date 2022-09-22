import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    """
    Calculates the Mean Absolute Error for our model
    max_leaf_nodes - The maximum number of nodes we want to search
    train_X - Features from training set
    val_X - Features from validation set
    train_y - target from training set
    val_y - target from validation set
    """
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

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

# Compare MAE with differing values of max_leaf_nodes
for max_leaf_node in [5, 50, 500, 5000]:
    temp = get_mae(max_leaf_node, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (max_leaf_node, 
                                                                temp))
