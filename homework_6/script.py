import pandas as pd
import seaborn as sns
import numpy as np

# %%
# imports
import matplotlib.pyplot as plt
# %%
# put problem statement in markdown
"""
10.2 Identifying Good System Administrators. A management consultant is studying
the roles played by experience and training in a system administrator’s ability to
complete a set of tasks in a specified amount of time. In particular, she is interested
in discriminating between administrators who are able to complete given tasks within
a specified time and those who are not. Data are collected on the performance of 75
randomly selected administrators. They are stored in the file SystemAdministrators.csv.
The variable Experience measures months of full-time system administrator experience,
while Training measures the number of relevant training credits. The outcome
variable Completed is either Yes or No, according to whether or not the administrator
completed the tasks.

a. Create a scatter plot of Experience vs. Training using color or symbol to distinguish
programmers who completed the task from those who did not complete it. Which
predictor(s) appear(s) potentially useful for classifying task completion?
b. Run a logistic regression model with both predictors using the entire dataset as
training data. Among those who completed the task, what is the percentage of
programmers incorrectly classified as failing to complete the task?
c. To decrease the percentage in part (b), should the cutoff probability be increased
or decreased?
d. How much experience must be accumulated by a programmer with 4 years of
training before his or her estimated probability of completing the task exceeds 0.5"
"""
# %%
# load homework_6/SystemAdministrators.csv
file_path = "SystemAdministrators.csv"
df = pd.read_csv(file_path)

# print column names as list
print(df.columns.tolist())

# for column print nans, unique values, and data types
for column in df.columns:
    print(f"{column}:")
    print(f"nans: {df[column].isnull().sum()}")
    print(f"unique values: {df[column].nunique()}")
    print(f"data type: {df[column].dtype}")
    print()

# %%
# set up to answer question a
sns.scatterplot(data=df, x="Experience", y="Training", hue="Completed task", palette=["red", "blue"])
plt.show()
# %%
# set up to answer question b
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# create X and y
X = df[["Experience", "Training"]]
y = df["Completed task"]

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create model
model = LogisticRegression()
model.fit(X_train, y_train)

# get predictions
y_pred = model.predict(X_test)

# get confusion matrix
confusion = confusion_matrix(y_test, y_pred)
print(confusion)

# Confusion matrix values
TN, FP, FN, TP = 10, 0, 1, 4

# Calculate the percentage of false negatives (incorrectly classified as not completing the task)
incorrect_percentage = (FN / (TP + FN)) * 100
incorrect_percentage

# %%
# set up to answer question c
# To decrease the percentage in part (b), should the cutoff probability be increased or decreased?
# The cutoff probability should be decreased to decrease the percentage of false negatives.

# %%
# set up to answer question d
# How much experience must be accumulated by a programmer with 4 years of training before his or her estimated probability of completing the task exceeds 0.5"
# create model
model = LogisticRegression()
model.fit(X, y)

# Get the coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# Print the coefficients and intercept
print("Coefficients:", coefficients)
print("Intercept:", intercept)

# %%
from scipy.special import expit  # Sigmoid function

# Coefficients and intercept from the logistic regression model
coefficients = [1.04592814, 0.1640162]
intercept = -10.24522925


# Function to calculate the estimated probability of completing the task
def estimate_probability(experience, training, coefficients, intercept):
    # Linear combination of inputs and coefficients
    linear_combination = (
        intercept + experience * coefficients[0] + training * coefficients[1]
    )
    # Apply the sigmoid function to get the probability
    probability = expit(linear_combination)
    return probability


# Function to find the required experience for a given level of training
# so that the estimated probability of completing the task exceeds 0.5
def required_experience_for_probability(
    training_months, threshold, coefficients, intercept
):
    experience_months = 0
    # Start at 0 months of experience and increment until the probability exceeds the threshold
    while True:
        probability = estimate_probability(
            experience_months, training_months, coefficients, intercept
        )
        if probability > threshold:
            return experience_months
        experience_months += 1


# Assuming 4 years of training which is 48 months
training_months = 48
threshold_probability = 0.5

# Calculate the required experience
required_experience = required_experience_for_probability(
    training_months, threshold_probability, coefficients, intercept
)
required_experience

# %%
# markdown with problem statement
"""
11.3 Car Sales. Consider the data on used cars (ToyotaCorolla.csv) with 1436 records and
details on 38 attributes, including Price, Age, KM, HP, and other specifications. The
goal is to predict the price of a used Toyota Corolla based on its specifications.
a. Fit a neural network model to the data. Use a single hidden layer with 2 nodes.
• Use predictors Age_08_04, KM, Fuel_Type, HP, Automatic, Doors, Quarterly_Tax,
Mfr_Guarantee, Guarantee_Period, Airco, Automatic_airco, CD_Player,
Powered_Windows, Sport_Model, and Tow_Bar.
• Remember to first scale the numerical predictor and outcome variables to a 0–1
scale (use function preprocess() with method = “range”—see Chapter 7) and convert
categorical predictors to dummies.
Record the RMS error for the training data and the validation data. Repeat
the process, changing the number of hidden layers and nodes to {single layer with
5 nodes}, {two layers, 5 nodes in each layer}.
i. What happens to the RMS error for the training data as the number of layers
and nodes increases?
ii. What happens to the RMS error for the validation data?
iii. Comment on the appropriate number of layers and nodes for this application."
"""
# %%
# load homework_6/ToyotaCorolla.csv
file_path = "ToyotaCorolla.csv"
df = pd.read_csv(file_path)

# print column names as list
print(df.columns.tolist())

# for column print nans, unique values, and data types
for column in df.columns:
    print(f"{column}:")
    print(f"nans: {df[column].isnull().sum()}")
    print(f"unique values: {df[column].nunique()}")
    print(f"data type: {df[column].dtype}")
    print()
# %%
# set up to answer question a
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# columns: ['Id', 'Model', 'Price', 'Age_08_04', 'Mfg_Month', 'Mfg_Year', 'KM', 'Fuel_Type', 'HP', 'Met_Color', 'Automatic', 'cc', 'Doors', 'Cylinders', 'Gears', 'Quarterly_Tax', 'Weight', 'Mfr_Guarantee', 'BOVAG_Guarantee', 'Guarantee_Period', 'ABS', 'Airbag_1', 'Airbag_2', 'Airco', 'Automatic_airco', 'Boardcomputer', 'CD_Player', 'Central_Lock', 'Powered_Windows', 'Power_Steering', 'Radio', 'Mistlamps', 'Sport_Model', 'Backseat_Divider', 'Metallic_Rim', 'Radio_cassette', 'Tow_Bar']

# Select the predictors and outcome variable
predictors = [
    "Age_08_04",
    "KM",
    "Fuel_Type",
    "HP",
    "Automatic",
    "Doors",
    "Quarterly_Tax",
    "Mfr_Guarantee",
    "Guarantee_Period",
    "Airco",
    "Automatic_airco",
    "CD_Player",
    "Powered_Windows",
    "Sport_Model",
    "Tow_Bar",
]

X = df[predictors]
y = df["Price"]

# Convert categorical predictors to dummies
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the numerical predictors and outcome variables to a 0–1 scale
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_valid_scaled = scaler_X.transform(X_valid)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_valid_scaled = scaler_y.transform(y_valid.values.reshape(-1, 1)).flatten()

# Create and fit the neural network model
model = MLPRegressor(hidden_layer_sizes=(2,), max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train_scaled)


# Function to train and evaluate the neural network
def train_evaluate_nn(hidden_layer_sizes):
    # Create the neural network model with specified hidden layer sizes
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes, max_iter=1000, random_state=42
    )

    # Fit the model to the scaled training data
    model.fit(X_train_scaled, y_train_scaled)

    # Predict on the training and validation data
    y_train_pred_scaled = model.predict(X_train_scaled)
    y_valid_pred_scaled = model.predict(X_valid_scaled)

    # Transform the predictions back to original scale
    y_train_pred = scaler_y.inverse_transform(
        y_train_pred_scaled.reshape(-1, 1)
    ).flatten()
    y_valid_pred = scaler_y.inverse_transform(
        y_valid_pred_scaled.reshape(-1, 1)
    ).flatten()

    # Calculate the RMS error for the training data
    rms_error_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

    # Calculate the RMS error for the validation data
    rms_error_valid = np.sqrt(mean_squared_error(y_valid, y_valid_pred))

    return rms_error_train, rms_error_valid


# Neural network configurations and RMS error calculations
rms_error_train_5_nodes, rms_error_valid_5_nodes = train_evaluate_nn((5,))
rms_error_train_5_5_nodes, rms_error_valid_5_5_nodes = train_evaluate_nn((5, 5))

# Output the RMS errors for each configuration
print(
    f"RMS error (single layer, 5 nodes) - Training: {rms_error_train_5_nodes}, Validation: {rms_error_valid_5_nodes}"
)
print(
    f"RMS error (two layers, 5 nodes each) - Training: {rms_error_train_5_5_nodes}, Validation: {rms_error_valid_5_5_nodes}"
)
# %%
# Assuming X_train_scaled, y_train_scaled, X_valid_scaled, y_valid_scaled are already defined
node_configs = [2, 4, 6, 8, 10]  # Example configurations to test
results = {}

for nodes in node_configs:
    model = MLPRegressor(hidden_layer_sizes=(nodes,), max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train_scaled)
    y_valid_pred_scaled = model.predict(X_valid_scaled)
    y_valid_pred = scaler_y.inverse_transform(
        y_valid_pred_scaled.reshape(-1, 1)
    ).flatten()
    rms_error_valid = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
    results[nodes] = rms_error_valid
    print(f"Nodes: {nodes}, Validation RMS Error: {rms_error_valid}")

# %%
