# %%
# imports
import os
import sys
import json
import requests
import datetime
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# %%
# turn off deprecated warnings
import warnings

warnings.filterwarnings("ignore")

# %%
# load data from homework_8/Coursetopics.csv
df = pd.read_csv("Coursetopics.csv")
print(df.head())
print(df.columns.tolist())

# %%
"""
Identifying Course Combinations. The Institute for Statistics Education at Statistics. com offers online courses in statistics and analytics, and is seeking information that will help in packaging and sequencing courses. Consider the data in the file Course-Topics.csv, the first few rows of which are shown in Table 14.13. These data are for purchases of online statistics courses at Statistics.com. Each row represents the courses attended by a single customer. The firm wishes to assess alternative sequencings and bundling of courses. Use association rules to analyze these data, and interpret several of the resulting rules.

['Intro',
 'DataMining',
 'Survey',
 'Cat Data',
 'Regression',
 'Forecast',
 'DOE',
 'SW']

 Data is already one-hot encoded
"""

# %%
# Exploratory Data Analysis
print(df.info())
print(df.describe())
print(df.isnull().sum())

# %%
# drop Intro column as it is not a course and will not be useful for the analysis
df = df.drop(columns=["Intro"])

# %%
# Set up association rules
# apriori algorithm

# Lower min_support to generate more complex itemsets
min_support = 0.009

while True:
    # Generate frequent itemsets with the current min_support
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    # Calculate the length of each itemset
    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))
    # Filter for itemsets that contain more than one course
    multi_course_itemsets = frequent_itemsets[frequent_itemsets["length"] > 1]

    # Check if the number of multi-course itemsets falls within the desired range
    if len(multi_course_itemsets) > 1 and len(multi_course_itemsets) < 500:
        break
    # Otherwise, adjust min_support and try again
    min_support *= 0.9
    # Break out of the loop if min_support becomes too low, to avoid infinite loops
    if min_support < 0.01:
        break

print(f"Final Minimum Support: {min_support}")
print("Frequent Multi-Course Itemsets:")
print(multi_course_itemsets)

# %%
from mlxtend.frequent_patterns import association_rules

# Generating association rules from the frequent itemsets
# You can adjust the metrics and thresholds according to your needs
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)

# Sorting the rules by descending confidence and lift for better insights
rules_sorted = rules.sort_values(["confidence", "lift"], ascending=[False, False])

print("Association Rules Sorted by Confidence and Lift:")
top_10 = (
    rules_sorted[["antecedents", "consequents", "support", "confidence", "lift"]]
    .head(10)
    .reset_index()
    .to_string()
)

# %%
# print a prettier looking table
print(top_10)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# plot the rules
fig, ax = plt.subplots(figsize=(20, 20))

# Add color gradient based on support
cmap = sns.cubehelix_palette(as_cmap=True)
points = ax.scatter(
    rules_sorted["confidence"],
    rules_sorted["lift"],
    c=rules_sorted["support"],
    s=100,
    cmap=cmap,
)

# Add color bar
fig.colorbar(points, ax=ax, label="Support")

# Add labels and arrows for top 10 rules
for i, rule in rules_sorted.head(10).iterrows():
    ax.annotate(
        f"{rule['antecedents']} -> {rule['consequents']}",
        xy=(rule["confidence"], rule["lift"]),
        xytext=(rule["confidence"] + 0.02, rule["lift"] + 0.02),
        fontsize=12,
        ha="center",
        va="center",
        color="black",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
        arrowprops=dict(arrowstyle="->", color="salmon"),
        zorder=10,
    )

# Add grid
ax.grid(True)

# Set title and labels
plt.title("Association Rules", fontsize=20)
plt.xlabel("Confidence", fontsize=16)
plt.ylabel("Lift", fontsize=16)

plt.show()
# %%
"""
15.1 University Rankings 

The dataset on American College and University Rankings contains information on 1302 American colleges and universities offering an undergraduate program. For each university, there are 17 measurements, including continuous measurements (such as tuition and graduation rate) and categorical measurements (such as location by state and whether it is a private or public school).
Note that many records are missing some measurements. Our first goal is to estimate these missing values from “similar” records. This will be done by clustering the complete records and then finding the closest cluster for each of the partial records. The missing values will be imputed from the information in that cluster.
Remove all records with missing measurements from the dataset.
For all the continuous measurements, run hierarchical clustering using complete linkage and Euclidean distance. Make sure to normalize the measurements. From the dendrogram: How many clusters seem reasonable for describing these data?
Compare the summary statistics for each cluster and describe each cluster in this context (e.g., “Universities with high tuition, low acceptance rate…”). Hint: To obtain cluster statistics for hierarchical clustering, use the aggregate() function.
Use the categorical measurements that were not used in the analysis (State and Private/Public) to characterize the different clusters. Is there any relationship between the clusters and the categorical information?
What other external information can explain the contents of some or all of these clusters?
Consider Tufts University, which is missing some information. Compute the Euclidean distance of this record from each of the clusters that you found above (using only the measurements that you have). Which cluster is it closest to? Impute the missing values for Tufts by taking the average of the cluster on those measurements.
"""
# %%
# load data from homework_8/Universities Dataset.csv
df = pd.read_csv("Universities Dataset.csv")
df_original = df.copy()
print(df.head())
print(df.columns.tolist())

# %%
# Exploratory Data Analysis
print(df.info())
print(df.describe())
print(df.isnull().sum())

# %%
# drop rows with missing values
df = df.dropna()
print(df.isnull().sum())

# %%
# Normalize the continuous measurements
from sklearn.preprocessing import StandardScaler

# Select the continuous measurements
continuous_columns = [
    col for col in df.columns if df[col].dtype in [np.float64, np.int64]
]

# print to confirm
print(continuous_columns)

# continuous cols are ['Public (1)/ Private (2)', "# appli. rec'd", '# appl. accepted', '# new stud. enrolled', '% new stud. from top 10%', '% new stud. from top 25%', '# FT undergrad', '# PT undergrad', 'in-state tuition', 'out-of-state tuition', 'room', 'board', 'add. fees', 'estim. book costs', 'estim. personal $', '% fac. w/PHD', 'stud./fac. ratio', 'Graduation rate']]

# norm df
norm_df = df.copy()

# Normalize the continuous measurements
scaler = StandardScaler()
norm_df[continuous_columns] = scaler.fit_transform(norm_df[continuous_columns])

# print to confirm
print(norm_df[continuous_columns].head())

# %%
# Hierarchical Clustering
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt

# Perform hierarchical clustering
Z = linkage(norm_df[continuous_columns], method="complete", metric="euclidean")

# Plot the dendrogram
plt.figure(figsize=(20, 10))
dendrogram(Z, labels=norm_df.index, leaf_rotation=90)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("University")
plt.ylabel("Distance")
plt.show()

# %%
# progammatically from this data determine and print the number of reasonable clusters
# Determine the number of reasonable clusters

# Set the threshold for defining clusters
threshold = 18

# Assign clusters to the data points
clusters = fcluster(Z, threshold, criterion="distance")

# Get the number of unique clusters
num_clusters = len(set(clusters))

# Print the number of reasonable clusters
print(f"Number of reasonable clusters: {num_clusters}")

# %%
# compare the summary statistics for each cluster
# and describe each cluster in this context
# Calculate the summary statistics for each cluster
cluster_stats = df.copy()
cluster_stats["cluster"] = clusters

# add to norm_df
norm_df["cluster"] = clusters

# Aggregate the data by cluster using aggregate()
# get numerical summary statistics for each cluster
num_cols = df.columns[df.dtypes.apply(lambda x: np.issubdtype(x, np.number))]
cluster_summary = cluster_stats.groupby("cluster")[num_cols].agg(
    ["mean", "std", "count"]
)

# Print the summary statistics for each cluster
print(cluster_summary)

# %%
# Use the categorical measurements that were not used in the analysis (State and Private/Public) to characterize the different clusters. Is there any relationship between the clusters and the categorical information?
# Add the categorical measurements to the cluster statistics

# I already added this categorical data back to the clusters by means of using the original non-normalized dataframe.

# What other external information can explain the contents of some or all of these clusters?
# The clusters could be further explained by external information such as the university's focus on specific fields of study, the size of the university, the location of the university, and the university's reputation.
# Consider Tufts University, which is missing some information. Compute the Euclidean distance of this record from each of the clusters that you found above (using only the measurements that you have). Which cluster is it closest to? Impute the missing values for Tufts by taking the average of the cluster on those measurements.

from scipy.spatial.distance import euclidean
from scipy.spatial import distance

tufts = df_original[
    df_original["College Name"].str.contains("Tufts", case=False, na=False)
]

# print to confirm
print(tufts)

# %%
# First, ensure 'Tufts University' data is a 1-D array for continuous columns
tufts_values = tufts[continuous_columns].dropna(axis=1).values.flatten()
# Assuming tufts_values, continuous_columns, and norm_df are defined as before
distances = []

# Extract the non-missing continuous columns for 'Tufts University'
non_missing_cols = tufts.dropna(axis=1).columns.intersection(continuous_columns)

for cluster_num in range(1, num_clusters + 1):
    # Compute the mean of the non-missing continuous columns for this cluster
    cluster_data = norm_df[norm_df["cluster"] == cluster_num]
    cluster_mean = cluster_data[non_missing_cols].mean().values

    # Ensure both tufts_values and cluster_mean are using the same set of columns
    tufts_non_missing_values = tufts[non_missing_cols].values.flatten()

    # Calculate Euclidean distance and append it to distances list
    dist = euclidean(tufts_non_missing_values, cluster_mean)
    distances.append(dist)

# Identify the closest cluster
closest_cluster = np.argmin(distances) + 1
print(f"Tufts University is closest to Cluster {closest_cluster}")

# Tufts University is closest to Cluster 2

# %%
# print columns in tufts that are missing values
missing_cols = tufts.columns[tufts.isnull().any()].tolist()

# print to confirm
print(missing_cols)

# %%
# Extract mean values for the closest cluster using ORIGINAL non-normalized data
cluster_means_original = cluster_data[cluster_data["cluster"] == closest_cluster][
    continuous_columns
].mean()

# Impute missing values in 'Tufts University' from the original dataframe
for col in missing_cols:
    if col in continuous_columns:  # Ensure we only impute continuous variables
        # Check if 'Tufts University' has a missing value in this column
        if pd.isnull(tufts[col].values[0]):
            # Impute using the mean from the closest cluster in the ORIGINAL data
            tufts.loc[:, col] = cluster_means_original[col]

print("Updated 'Tufts University' record with imputed values:")
print(tufts.loc[:, continuous_columns])

# %%
# Impute the missing values
for col in missing_cols:
    if col in non_missing_cols:
        continue  # Skip non-missing columns
    # Impute missing values using the mean from the closest cluster
    mean_val = norm_df[norm_df["cluster"] == closest_cluster][col].mean()
    tufts.loc[:, col] = mean_val

print("Updated 'Tufts University' record with imputed values:")
print(tufts[continuous_columns])

# %%
