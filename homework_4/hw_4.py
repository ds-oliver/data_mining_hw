# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


# %%
# load "BostonHousing.csv" into a DataFrame
df = pd.read_csv("BostonHousing.csv")

# The file BostonHousing.csv contains information
# collected by the US Bureau of the Census concerning housing in the area of
# Boston, Massachusetts. The dataset includes information on 506 census housing tracts
# in the Boston area. The goal is to predict the median house price in new tracts based
# on information such as crime rate, pollution, and number of rooms. The dataset contains
# 13 predictors, and the response is the median house price (MEDV).

## Fit a multiple linear regression model to the median house price (MEDV) as a function of CRIM, CHAS, and RM. Write the equation for predicting the median house price from the predictors in the model.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Define the predictors and target variable
X = df[["CRIM", "CHAS", "RM"]]
y = df["MEDV"]

# Split the data into training and validation sets (70% train, 30% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit a multiple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

coefficients, intercept

print(f"MEDV = {intercept} + {coefficients[0]} * CRIM + {coefficients[1]} * CHAS + {coefficients[2]} * RM")
# %%
## Using the estimated regression model, what median house price is predicted for a tract in the Boston area that does not bound the Charles River, has a crime rate of 0.1, and where the average number of rooms per house is 6? What is the prediction error?
# Create a new data point with the given values
new_data = {"CRIM": 0.1, "CHAS": 0, "RM": 6}

# Predict the median house price
predicted_price = model.predict([list(new_data.values())])

# Compute the prediction error
actual_price = df["MEDV"].mean()
prediction_error = actual_price - predicted_price

predicted_price, prediction_error

print(f"The predicted median house price is ${predicted_price[0]:.2f}. The prediction error is {prediction_error[0]:.2f}.")

# %%
# Next steps involve reducing the number of predictors by analyzing the relationships among INDUS, NOX, and TAX, computing the correlation table for the numerical predictors to identify redundancy, and performing stepwise regression to find the best model.
# Select only the numeric columns from the DataFrame
df_numeric = df.select_dtypes(include=[np.number])

# Compute the correlation table for only the numeric columns
correlation = df_numeric.corr()

import seaborn as sns

# Generate a heatmap
sns.heatmap(correlation, annot=True, cmap='coolwarm')

# decrease font size
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

# Display the plot
plt.show()
# %%
# Use stepwise regression with the three options (backward, forward, both) to reduce the remaining predictors as follows: Run stepwise on the training set. Choose the top model from each stepwise run. Then use each of these models separately to predict the validation set. Compare RMSE, MAPE, and mean error, as well as lift charts. Finally, describe the best model.
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Define the predictors and target variable
X = df.drop(columns=["MEDV"])
y = df["MEDV"]

# Split the data into training and validation sets (70% train, 30% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Stepwise regression

# Backward stepwise regression

# Add a constant to the predictors
X_train = add_constant(X_train)

# Fit the model
model = sm.OLS(y_train, X_train).fit()

# Perform backward stepwise regression
def backward_stepwise_selection(X, y):
    # Create a list to store the predictors to remove
    to_remove = []
    
    # Perform backward stepwise regression
    for i in range(X.shape[1]):
        # Add a constant to the predictors
        X = add_constant(X)
        
        # Fit the model
        model = sm.OLS(y, X).fit()
        
        # Find the predictor with the highest p-value
        max_p_value = model.pvalues.drop("const").max()
        max_p_value_index = model.pvalues.drop("const").idxmax()
        
        # If the highest p-value is greater than 0.05, remove the predictor
        if max_p_value > 0.05:
            to_remove.append(max_p_value_index)
            X = X.drop(columns=max_p_value_index)
        else:
            break
    
    return X

# Perform backward stepwise regression
X_train_backward = backward_stepwise_selection(X_train, y_train)

# Add the "const" column to X_val if it is not present
if "const" not in X_val.columns:
    X_val = add_constant(X_val)

# Add missing columns from X_train_backward to X_val_backward
missing_columns = set(X_train_backward.columns) - set(X_val.columns)
X_val_backward = X_val.reindex(columns=X_val.columns.tolist() + list(missing_columns))

# Predict the validation set
y_pred_backward = model.predict(X_val_backward)

# Compute the RMSE, MAPE, and mean error
rmse_backward = np.sqrt(mean_squared_error(y_val, y_pred_backward))
mape_backward = np.mean(np.abs((y_val - y_pred_backward) / y_val)) * 100
mean_error_backward = np.mean(y_val - y_pred_backward)

# Forward stepwise regression
# Perform forward stepwise regression
def forward_stepwise_selection(X, y):
    # Create a list to store the predictors to include
    to_include = []
    
    # Perform forward stepwise regression
    for i in range(X.shape[1]):
        # Add a constant to the predictors
        X = add_constant(X)
        
        # Fit the model
        model = sm.OLS(y, X).fit()
        
        # Find the predictor with the lowest p-value
        min_p_value = model.pvalues.drop("const").min()
        min_p_value_index = model.pvalues.drop("const").idxmin()
        
        # If the lowest p-value is less than 0.05, include the predictor
        if min_p_value < 0.05:
            to_include.append(min_p_value_index)
            X = X[to_include]
        else:
            break
    
    return X

# Perform forward stepwise regression
X_train_forward = forward_stepwise_selection(X_train, y_train)

# Add the "const" column to X_val if it is not present
if "const" not in X_val.columns:
    X_val = add_constant(X_val)

# Fit the model
model = sm.OLS(y_train, X_train_forward).fit()

# Predict the validation set
X_val_forward = X_val[X_train_forward.columns]
y_pred_forward = model.predict(X_val_forward)

# Compute the RMSE, MAPE, and mean error
rmse_forward = np.sqrt(mean_squared_error(y_val, y_pred_forward))
mape_forward = np.mean(np.abs((y_val - y_pred_forward) / y_val)) * 100
mean_error_forward = np.mean(y_val - y_pred_forward)


# Predict the validation set
X_val_forward = X_val[X_train_forward.columns]
X_val_forward = add_constant(X_val_forward)
y_pred_forward = model.predict(X_val_forward)

# Compute the RMSE, MAPE, and mean error
rmse_forward = np.sqrt(mean_squared_error(y_val, y_pred_forward))
mape_forward = np.mean(np.abs((y_val - y_pred_forward) / y_val)) * 100
mean_error_forward = np.mean(y_val - y_pred_forward)

# Both stepwise regression
# Perform both stepwise regression
def both_stepwise_selection(X, y):
    # Create a list to store the predictors to include
    to_include = []
    
    # Perform both stepwise regression
    for i in range(X.shape[1]):
        # Add a constant to the predictors
        X = add_constant(X)
        
        # Fit the model
        model = sm.OLS(y, X).fit()
        
        # Find the predictor with the lowest p-value
        min_p_value = model.pvalues.drop("const").min()
        min_p_value_index = model.pvalues.drop("const").idxmin()
        
        # If the lowest p-value is less than 0.05, include the predictor
        if min_p_value < 0.05:
            to_include.append(min_p_value_index)
            X = X[to_include]
        else:
            break
    
    return X

# Perform both stepwise regression
X_train_both = both_stepwise_selection(X_train, y_train)

# Fit the model
model = sm.OLS(y_train, X_train_both).fit()

# Predict the validation set
X_val_both = X_val[X_train_both.columns]
X_val_both = add_constant(X_val_both)
y_pred_both = model.predict(X_val_both)

# Compute the RMSE, MAPE, and mean error
rmse_both = np.sqrt(mean_squared_error(y_val, y_pred_both))
mape_both = np.mean(np.abs((y_val - y_pred_both) / y_val)) * 100
mean_error_both = np.mean(y_val - y_pred_both)

# Compare the results
rmse_backward, mape_backward, mean_error_backward, rmse_forward, mape_forward, mean_error_forward, rmse_both, mape_both, mean_error_both

print(f"Backward stepwise regression: RMSE = {rmse_backward:.2f}, MAPE = {mape_backward:.2f}%, mean error = {mean_error_backward:.2f}")
print(f"Forward stepwise regression: RMSE = {rmse_forward:.2f}, MAPE = {mape_forward:.2f}%, mean error = {mean_error_forward:.2f}")
print(f"Both stepwise regression: RMSE = {rmse_both:.2f}, MAPE = {mape_both:.2f}%, mean error = {mean_error_both:.2f}")

# results: Backward stepwise regression: RMSE = 3.72, MAPE = 13.39%, mean error = 0.05; Forward stepwise regression: RMSE = 5.46, MAPE = 26.72%, mean error = -0.55; Both stepwise regression: RMSE = 5.46, MAPE = 26.72%, mean error = -0.55
print("The best model is the backward stepwise regression model, which has the lowest RMSE and MAPE, and the smallest mean error.")
# %%
## 7.3
# Predicting Housing Median Prices. The file BostonHousing.csv contains information on over 500 census tracts in Boston, where for each tract multiple variables are recorded. The last column (CAT.MEDV) was derived from MEDV, such that it obtains the value 1 if MEDV > 30 and 0 otherwise. Consider the goal of predicting the median value (MEDV) of a tract, given the information in the first 12 columns. Partition the data into training (60%) and validation (40%) sets.
# Perform a k-NN prediction with all 12 predictors (ignore the CAT.MEDV column), trying values of k from 1 to 5. Make sure to normalize the data, and choose function knn() from package class rather than package FNN. To make sure R is using the class package (when both packages are loaded), use class::knn(). What is the best k? What does it mean?
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

df = pd.read_csv("BostonHousing.csv")

# Define the predictors and target variable
X = df.drop(columns=["CAT. MEDV"])
y = df["MEDV"]

# Split the data into training and validation sets (60% train, 40% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)

# Fit a k-NN model for k = 1 to 5
k_values = [1, 2, 3, 4, 5]
rmse_values = []

for k in k_values:
    # Fit a k-NN model
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train_normalized, y_train)
    
    # Normalize the validation set
    X_val_normalized = scaler.transform(X_val)
    
    # Predict the validation set
    y_pred = model.predict(X_val_normalized)
    
    # Compute the RMSE
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    rmse_values.append(rmse)

# Find the best k
best_k = k_values[np.argmin(rmse_values)]
best_k

# Predict the MEDV for a tract with the following information, using the best k:
# i am inserting tract info as the assignment did not provide the values, i will use random values within the range of the dataset
tract_info = {}
# get the range of each column to generate random values
for col in X.columns:
    # get the range of each column
    min_value = X[col].min()
    max_value = X[col].max()
    # generate a random value within the range of each column
    random_value = np.random.uniform(min_value, max_value)
    tract_info[col] = random_value

print(tract_info)

# Fit a k-NN model with the best k
model = KNeighborsRegressor(n_neighbors=best_k)
model.fit(X_train_normalized, y_train)

# Normalize the tract info
tract_info_normalized = scaler.transform([list(tract_info.values())])

# Predict the MEDV
predicted_medv = model.predict(tract_info_normalized)
predicted_medv

print(f"The predicted MEDV for the tract is {predicted_medv[0]:.2f}.")

# If we used the above k-NN algorithm to score the training data, what would be the error of the training set?
# Predict the training set
y_pred_train = model.predict(X_train_normalized)

# Compute the RMSE
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_train

print(f"The RMSE of the training set is {rmse_train:.2f}.")

# The RMSE of the training set is 1.63.

# %%
