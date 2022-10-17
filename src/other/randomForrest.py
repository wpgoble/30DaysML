import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

file_src = '../data/melb_data.csv'
df = pd.read_csv(file_src)

df = df.dropna(axis=0)
y = df.Price
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = df[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)

melb_predict = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_predict))