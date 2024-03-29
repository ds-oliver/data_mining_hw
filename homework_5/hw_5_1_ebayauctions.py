# %%
# imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree, export_text
from sklearn.metrics import roc_curve, auc

# %%
# load homework_5/eBayAuctions.csv
file_path = "eBayAuctions.csv"
df = pd.read_csv(file_path)

# %%
# do a bit of exploratory data analysis
print(df.dtypes)
print(df.head())
print(df.describe())
print(df.isnull().sum())
print(df.nunique())

# %%
# Convert 'Duration' into a categorical variable
df["Duration"] = df["Duration"].astype("category")

# Handling categorical variables by encoding them
label_encoders = {}
for column in df.select_dtypes(include=["object", "category"]).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# %%
# Partition the data into training (60%) and validation (40%) sets
df_train, df_test = train_test_split(df, test_size=0.4, random_state=42)
# %%
# Decision Tree Classifier
# Set the minimum number of records in a terminal node to 50
min_samples_leaf = 50
# Set the maximum number of levels to be displayed at seven
max_depth = 7

# Create the decision tree classifier
clf = DecisionTreeClassifier(
    min_samples_leaf=min_samples_leaf, max_depth=max_depth, random_state=42
)

# The target variable is 'Competitive?'
y = df_train["Competitive?"]
X = df_train.drop("Competitive?", axis=1)

# Fit the classifier to the training data
clf.fit(X, y)

# Assuming clf is your trained DecisionTreeClassifier model
rules = export_text(clf, feature_names=list(X.columns))
print(rules)


# %%
# Display the decision tree
plt.figure(figsize=(30, 10))
plot_tree(
    clf,
    filled=True,
    feature_names=list(X.columns),
    class_names=["Non-Competitive", "Competitive"],
    rounded=True,
    fontsize=12,
)
plt.show()


# %%
# Splitting the dataset into training and testing sets
categorical_features = [
    "Category",
    "Duration",
    "endDay",
    "currency",
]  # Assuming 'currency' and 'endDay' are also pre-auction predictors
label_encoders = {}
for column in categorical_features:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Splitting the dataset into training and testing sets
X = df[
    ["OpenPrice", "sellerRating", "Category", "Duration"]
]  # Add or remove predictors as necessary
y = df["Competitive?"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# Initialize and fit the Decision Tree Classifier
clf = DecisionTreeClassifier(min_samples_leaf=50, max_depth=7, random_state=42)
clf.fit(X_train, y_train)

# Print the decision rules
rules = export_text(clf, feature_names=list(X.columns))
print(rules)

# Optionally, visualize the tree
plt.figure(figsize=(30, 10))
plot_tree(
    clf,
    filled=True,
    feature_names=list(X.columns),
    class_names=["Non-Competitive", "Competitive"],
    rounded=True,
    fontsize=12,
)
plt.show()

# %%
# to visualize this on a scatter plot, you'll want to plot OpenPrice on one axis and sellerRating on the other since these seem to be the most significant predictors based on the tree structure. Competitiveness would be denoted by color. You'd draw vertical lines at OpenPrice splits (e.g., 3.62, 11.28) and horizontal lines at sellerRating splits (e.g., 601.5, 2365.5).
# Create a scatter plot
plt.figure(figsize=(10, 10))
sns.scatterplot(
    x="OpenPrice",
    y="sellerRating",
    hue="Competitive?",
    data=df,
    palette=["blue", "red"],
    alpha=0.5,
)

# draw vertical lines at OpenPrice splits
plt.axvline(x=3.62, color="gold", linestyle="--")
plt.axvline(x=11.28, color="gold", linestyle="--")


# %%
# lift chart and confusion matrix
# Create a lift chart
plt.figure(figsize=(10, 10))
sns.histplot(
    x="OpenPrice",
    hue="Competitive?",
    data=df,
    kde=True,
    stat="probability",
    common_norm=False,
    palette=["blue", "red"],
    alpha=0.5,
)

# generate a lift chart
plt.axhline(y=0.5, color="black", linestyle="--")
plt.axhline(y=0.25, color="black", linestyle="--")
plt.axhline(y=0.75, color="black", linestyle="--")
plt.axvline(x=3.62, color="gold", linestyle="--")
plt.axvline(x=11.28, color="gold", linestyle="--")


# %%
y_test = y_test.reset_index(drop=True)

# Predict probabilities on the validation set
y_pred_probs = clf.predict_proba(X_test)[:, 1]

# Sort the probabilities and the true values based on the probabilities
sorted_indices = np.argsort(y_pred_probs)[::-1]
sorted_probs = y_pred_probs[sorted_indices]
sorted_true_values = y_test.iloc[sorted_indices]

# Calculate the lift curve
gains = np.cumsum(sorted_true_values) / sum(y_test)
gains = np.insert(gains, 0, 0)  # Insert a 0 at the beginning of the gains array

# Calculate the percentage of all data
percentage_of_data = np.linspace(0, 1, len(gains))

# Plot the lift chart
plt.figure(figsize=(10, 6))
plt.plot(percentage_of_data, gains, drawstyle="steps-post", label="Lift Curve")
plt.plot(percentage_of_data, percentage_of_data, label="Baseline", linestyle="--")

plt.xlabel("Percentage of data")
plt.ylabel("Cumulative gain")
plt.title("Lift Chart")
plt.legend()
plt.grid()
plt.show()


# %%
# Create a confusion matrix
from sklearn.metrics import confusion_matrix

# Predict the target variable for the testing set
y_pred = clf.predict(X_test)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
print(cm)

# %%
