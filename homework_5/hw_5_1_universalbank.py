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

# %%
# load homework_5/eBayAuctions.csv
file_path = "UniversalBank.csv"
df = pd.read_csv(file_path)

# %%
# do a bit of exploratory data analysis
print(df.dtypes)
print(df.head())
print(df.describe())
print(df.isnull().sum())
print(df.nunique())

# %%
# Partition the data into training (60%) and validation (40%) sets
df_train, df_test = train_test_split(df, test_size=0.4, random_state=42)

# Create a pivot table for the training data
pivot_table = pd.pivot_table(
    df_train,
    index=["CreditCard", "Personal Loan"],
    columns="Online",
    aggfunc="size",
    fill_value=0,
)

# what is the probability that this customer will accept the loan offer? [This is the probability of loan acceptance (Loan = 1) conditional on having a bank credit card  (CC = 1) and being an active user of online banking services (Online = 1)].

probability = pivot_table.loc[(1, 1), 1] / pivot_table.loc[(1, slice(None)), 1].sum()

# Print the probability
print(f"The probability of loan acceptance is {probability:.2%}")

# %%
# Create two separate pivot tables for the training data. One will have Loan (rows) as a function of Online (columns) and the other will have Loan (rows) as a function of CC.
pivot_table_online = pd.pivot_table(
    df_train,
    index="Personal Loan",
    columns="Online",
    aggfunc="size",
    fill_value=0,
)


pivot_table_cc = pd.pivot_table(
    df_train,
    index="Personal Loan",
    columns="CreditCard",
    aggfunc="size",
    fill_value=0,
)
# Compute the following quantities
# P(CC = 1 j Loan = 1) (the proportion of credit card holders among the loan acceptors)
pivot_table_cc = pivot_table_cc.div(pivot_table_cc.sum(axis=1), axis=0)

# P(Online = 1 j Loan = 1) (the proportion of online banking users among the loan acceptors)
pivot_table_online = pivot_table_online.div(pivot_table_online.sum(axis=1), axis=0)

# P(Loan = 1) (the proportion of loan acceptors)
p_loan = df_train["Personal Loan"].mean()

# P(CC = 1 j Loan = 0)
p_cc_no_loan = pivot_table_cc.loc[0, 1]

# P(Online = 1 j Loan = 0)
p_online_no_loan = pivot_table_online.loc[0, 1]

# P(Loan = 0)
p_no_loan = 1 - p_loan

# print results as fstrings
print(f"P(CC = 1 | Loan = 1) = {pivot_table_cc.loc[1, 1]:.2%}")

print(f"P(Online = 1 | Loan = 1) = {pivot_table_online.loc[1, 1]:.2%}")

print(f"P(Loan = 1) = {p_loan:.2%}")

print(f"P(CC = 1 | Loan = 0) = {p_cc_no_loan:.2%}")

print(f"P(Online = 1 | Loan = 0) = {p_online_no_loan:.2%}")

print(f"P(Loan = 0) = {p_no_loan:.2%}")

# %%
# Use the quantities computed above to compute the naive Bayes probability P(Loan = 1 j CC = 1, Online = 1).
# Calculate unconditional probabilities for the denominator of the naive Bayes formula
p_cc = df_train["CreditCard"].mean()  # P(CC = 1)
p_online = df_train["Online"].mean()  # P(Online = 1)

# Naive Bayes probability calculation using the correct unconditional probabilities
p_loan_given_cc_online_correct = (
    pivot_table_cc.loc[1, 1] * pivot_table_online.loc[1, 1] * p_loan
) / (p_cc * p_online)

# Print the corrected probability
print(f"The probability of loan acceptance is {p_loan_given_cc_online_correct:.2%}")

# %%
