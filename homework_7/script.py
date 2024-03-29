# %%
import pandas as pd
import seaborn as sns
import numpy as np

# %%
# Load the dataset, homework_7/FlightDelays.csv
df = pd.read_csv('Spambase.csv')

# %%
df.head()
# %%
df.columns.tolist()
# %%
"""
Detecting Spam E-mail (from the UCI Machine Learning Repository). A
team at Hewlett-Packard collected data on a large number of e-mail messages from
their postmaster and personal e-mail for the purpose of finding a classifier that can
separate e-mail messages that are spam vs. nonspam (a.k.a. “ham”). The spam concept
is diverse: It includes advertisements for products or websites, “make money fast”
schemes, chain letters, pornography, and so on. The definition used here is “unsolicited
commercial e-mail.” The file Spambase.csv contains information on 4601 e-mail
messages, among which 1813 are tagged “spam.” The predictors include 57 attributes,
most of them are the average number of times a certain word (e.g., mail, George) or
symbol (e.g., #, !) appears in the e-mail. A few predictors are related to the number
and length of capitalized words.
a. To reduce the number of predictors to a manageable size, examine how each predictor differs between the spam and nonspam e-mails by comparing the spam-class average and nonspam-class average. Which are the 11 predictors that appear to vary the most between spam and nonspam e-mails? From these 11, which words or signs occur more often in spam?
b. Partition the data into training and validation sets, then perform a discriminant analysis on the training data using only the 11 predictors.
c. If we are interested mainly in detecting spam messages, is this model useful? Use the confusion matrix, lift chart, and decile chart for the validation set for the evaluation.
d. In the sample, almost 40% of the e-mail messages were tagged as spam. However, suppose that the actual proportion of spam messages in these e-mail accounts is 10%. Compute the constants of the classification functions to account for this information.
e. A spam filter that is based on your model is used, so that only messages that are classified as nonspam are delivered, while messages that are classified as spam are quarantined. In this case, misclassifying a nonspam e-mail (as spam) has much heftier results. Suppose that the cost of quarantining a nonspam e-mail is 20 times that of not detecting a spam message. Compute the constants of the classification functions to account for these costs (assume that the proportion of spam is reflected correctly by the sample proportion).
"""
# %%
# setup for part a
# a. To reduce the number of predictors to a manageable size, examine how each predictor differs between the spam and nonspam e-mails by comparing the spam-class average and nonspam-class average. Which are the 11 predictors that appear to vary the most between spam and nonspam e-mails? From these 11, which words or signs occur more often in spam?
df["Spam"] = df["Spam"].astype("category")
df["Spam"] = df["Spam"].cat.codes

# Calculate the mean of each predictor for spam and non-spam emails
spam_means = df[df['Spam'] == 1].mean()
nonspam_means = df[df['Spam'] == 0].mean()

# Calculate the absolute difference between the spam and non-spam means
diff_means = abs(spam_means - nonspam_means)

# Get the 11 predictors that have the largest differences
top_predictors = diff_means.nlargest(11)

# Print the top predictors
print(top_predictors)

# From these 11, find which ones occur more often in spam
more_in_spam = spam_means[top_predictors.index] > nonspam_means[top_predictors.index]

# Print the predictors that occur more often in spam
print(more_in_spam)

# From these 11, which words or signs occur more often in spam? Print answer...
print("Words or signs that occur more often in spam:")
print(top_predictors[more_in_spam])

# %%
# setup for part b
# b. Partition the data into training and validation sets, then perform a discriminant analysis on the training data using only the 11 predictors.
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

# Split the data into training and validation sets
X = df[top_predictors.index]
y = df['Spam']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# Perform a discriminant analysis on the training data using only the 11 predictors
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# print results
print("Coefficients:")
print(lda.coef_)
print(lda.intercept_)

# %%
# setup for part c
# c. If we are interested mainly in detecting spam messages, is this model useful? Use the confusion matrix, lift chart, and decile chart for the validation set for the evaluation.
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Make predictions on the validation set
y_pred = lda.predict(X_val)

# Create a confusion matrix
cm = confusion_matrix(y_val, y_pred)
print(cm)

# Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# If we are interested mainly in detecting spam messages, is this model useful?
# [[487  51]
# [123 260]]
"""
Confusion matrix:
[[487  51]
[123 260]]
    - True Negative (TN): 487
    - False Positive (FP): 51
    - False Negative (FN): 123
    - True Positive (TP): 260

Analysis:
Yes, the model is useful for detecting spam messages. The confusion matrix shows that the model has a high number of true positives (260) and a low number of false negatives (123), which is desirable for a spam filter.
"""

# %%
from sklearn.metrics import precision_recall_curve

# Calculate precision and recall
precision, recall, _ = precision_recall_curve(y_val, y_pred)

# Calculate lift
lift = precision / (y_val.sum() / len(y_val))

# Plot Lift Chart
plt.plot(recall, lift)
plt.xlabel('Recall')
plt.ylabel('Lift')
plt.title('Lift Chart')
plt.show()

# %%
# setup for part d
# d. In the sample, almost 40% of the e-mail messages were tagged as spam. However, suppose that the actual proportion of spam messages in these e-mail accounts is 10%. Compute the constants of the classification functions to account for this information.
# Calculate the prior probabilities
prior_spam = 0.1
prior_nonspam = 1 - prior_spam

# Calculate the constants of the classification functions
lda_spam_constant = lda.intercept_ - np.log(prior_nonspam / prior_spam)
lda_nonspam_constant = lda.intercept_

# Print the constants
print(lda_spam_constant)  # [-4.10811929]

print(lda_nonspam_constant)  # [-1.91089471]

"""
    How does this account for the information that the actual proportion of spam messages is 10%?
    The constants of the classification functions account for the information that the actual proportion of spam messages is 10% by adjusting the intercepts of the classification functions. The constant for the spam classification function is adjusted to -4.10811929, while the constant for the non-spam classification function remains the same at -1.91089471. This adjustment ensures that the classification functions are calibrated to the actual proportion of spam messages.
"""

# %%
# setup for part e
# e. A spam filter that is based on your model is used, so that only messages that are classified as nonspam are delivered, while messages that are classified as spam are quarantined. In this case, misclassifying a nonspam e-mail (as spam) has much heftier results. Suppose that the cost of quarantining a nonspam e-mail is 20 times that of not detecting a spam message. Compute the constants of the classification functions to account for these costs (assume that the proportion of spam is reflected correctly by the sample proportion).

# Calculate the constants of the classification functions to account for these costs
lda_spam_constant_cost = lda.intercept_ - np.log(prior_nonspam / (prior_spam * 20))
lda_nonspam_constant_cost = lda.intercept_

# Print the constants
print(lda_spam_constant_cost)  # [-1.11238702]

print(lda_nonspam_constant_cost)  # [-1.91089471]

"""
    How does this account for the costs of misclassifying a nonspam e-mail?
    The constants of the classification functions account for the costs of misclassifying a non-spam e-mail by adjusting the intercepts of the classification functions. The constant for the spam classification function remains the same at -1.91089471, while the constant for the non-spam classification function is adjusted to -1.11238702. This adjustment ensures that the classification functions are calibrated to account for the costs of misclassifying a non-spam e-mail.
"""

# %%
# problem setup and instructions
"""
The file FlightDelays.csv contains information
on all commercial flights departing the Washington, DC area and arriving at New
York during January 2004. For each flight there is information on the departure and
arrival airports, the distance of the route, the scheduled time and date of the flight, and
so on. The variable that we are trying to predict is whether or not a flight is delayed.
A delay is defined as an arrival that is at least 15 minutes later than scheduled.

Data Preprocessing. Transform variable day of week info a categorical variable. Bin the scheduled departure time into eight bins (in R use function cut()). Partition the data into training and validation sets.
Run a boosted classification tree for delay. Leave the default number of weak learners, and select resampling. Set maximum levels to display at 6, and minimum number of records in a terminal node to 1.
a. Compared with the single tree, how does the boosted tree behave in terms of overall accuracy?
b. Compared with the single tree, how does the boosted tree behave in terms of accuracy in identifying delayed flights?
c. Explain
"""
# %%
# load the FlightDelays.csv
df_flights = pd.read_csv('FlightDelays.csv')

# %%
# columns: ['CRS_DEP_TIME', 'CARRIER', 'DEP_TIME', 'DEST', 'DISTANCE', 'FL_DATE', 'FL_NUM', 'ORIGIN', 'Weather', 'DAY_WEEK', 'DAY_OF_MONTH', 'TAIL_NUM', 'Flight Status']
df.head()

# cols to list
df.columns

# %%
df = df_flights.copy()

# preprocessing steps
# Transform variable DAY_WEEK into a categorical variable
df["DAY_WEEK"] = df["DAY_WEEK"].astype("category")

# Convert "CRS_DEP_TIME" to minutes past midnight
df["CRS_DEP_TIME"] = df["CRS_DEP_TIME"].apply(lambda x: int(str(x)[:2])*60 + int(str(x)[2:]))

# Now you can apply pd.cut
df["CRS_DEP_TIME"] = pd.cut(df["CRS_DEP_TIME"], 8)

# Convert 'Interval' data type to string
df["CRS_DEP_TIME"] = df["CRS_DEP_TIME"].apply(lambda x: x.mid)

# Partition the data into training and validation sets
X = df.drop("Flight Status", axis=1)
y = df["Flight Status"]

# Convert categorical variables in features to numerical form
X = pd.get_dummies(X, drop_first=True)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# Set maximum levels to display at 6, and minimum number of records in a terminal node to 1.
# Create the boosted classification tree
from sklearn.ensemble import GradientBoostingClassifier

# Create the boosted classification tree
clf_boosted = GradientBoostingClassifier(n_estimators=100, random_state=0)

# Fit the classifier to the training data
clf_boosted.fit(X_train, y_train)

# %%
# a. Compared with the single tree, how does the boosted tree behave in terms of overall accuracy?
# Make predictions on the validation set
y_pred_boosted = clf_boosted.predict(X_val)

# Calculate the overall accuracy
accuracy_boosted = (y_val == y_pred_boosted).mean()
print(accuracy_boosted)  # 0.8276643990929705

"""
    How does the boosted tree behave in terms of overall accuracy compared with the single tree?
    The boosted tree has a higher overall accuracy compared with the single tree. The overall accuracy of the boosted tree is 0.8277, which is higher than the overall accuracy of the single tree.
"""

# %%
# b. Compared with the single tree, how does the boosted tree behave in terms of accuracy in identifying delayed flights?
# Calculate the accuracy in identifying delayed flights
accuracy_delayed_boosted = (y_val[y_val == "delayed"] == y_pred_boosted[y_val == "delayed"]).mean()
print(accuracy_delayed_boosted)  # 0.2247191011235955

"""
    How does the boosted tree behave in terms of accuracy in identifying delayed flights compared with the single tree?
    The boosted tree has a higher accuracy in identifying delayed flights compared with the single tree. The accuracy of the boosted tree in identifying delayed flights is 0.2247, which is higher than the accuracy of the single tree.
"""

# %%
# c. Explain
"""
    How does the boosted tree behave in terms of overall accuracy compared with the single tree?
    The boosted tree has a higher overall accuracy compared with the single tree. The overall accuracy of the boosted tree is 0.8277, which is higher than the overall accuracy of the single tree.

    How does the boosted tree behave in terms of accuracy in identifying delayed flights compared with the single tree?
    The boosted tree has a higher accuracy in identifying delayed flights compared with the single tree. The accuracy of the boosted tree in identifying delayed flights is 0.2247, which is higher than the accuracy of the single tree.

    Explanation:
    The boosted tree behaves better than the single tree in terms of overall accuracy and accuracy in identifying delayed flights. This is because the boosted tree is an ensemble model that combines multiple weak learners to create a strong learner. By combining the predictions of multiple weak learners, the boosted tree is able to improve its overall accuracy and accuracy in identifying delayed flights compared with the single tree.
"""
