import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# read data
dataframe = pd.read_table("challenge_dataset.txt", sep=",", names=('A', 'B'))
# dataframe = pd.read_fwf('brain_body.txt')
x_values = dataframe[["A"]]
y_values = dataframe[["B"]]

x_train, x_test, y_train, y_test = train_test_split(
    x_values, y_values, test_size=0.5, random_state=42)
# train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_train, y_train)
print body_reg.score(x_test, y_test)
pred = body_reg.predict(x_test)
# visualize results
plt.scatter(x_test, y_test)
plt.plot(x_test, pred)
plt.show()
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, pred))