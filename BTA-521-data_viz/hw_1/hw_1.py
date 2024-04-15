# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so

# %%
# load hw_1/SupermarketTransactions.xlsx
supermarket_data = pd.read_excel("SupermarketTransactions.xlsx")
supermarket_data.head()

# %%
# get columns as list
columns = supermarket_data.columns.tolist()
columns

# %%
# i. Number of transactions
num_transactions = len(supermarket_data)
print(f"Number of transactions: {num_transactions}")

# ii. Number of unique customers
num_customers = supermarket_data["Customer"].nunique()
print(f"Number of customers: {num_customers}")

# iii. Number of different dates
num_dates = supermarket_data["Purchase"].nunique()
print(f"Number of different dates: {num_dates}")

# %%
# Bar chart of Country
(
    so.Plot(supermarket_data, "Country")
    .add(so.Bar(), so.Count())
    .label(x="Country", y="Number of Transactions")
    .show()
)

# What do you conclude about the relationship of Country with shopping at these stores?
# Most of the transactions are from the United States. This might be due to the fact that most of the supermarkets are located in the United States.

# %%
# ring chart of Country using so.Plot
# Continue using country_counts as defined above
# Calculate counts for each country, state, and city if not already done
country_counts = supermarket_data["Country"].value_counts().reset_index()
country_counts.columns = ["Country", "Counts"]

state_counts = supermarket_data["State"].value_counts().reset_index()
state_counts.columns = ["State", "Counts"]

city_counts = supermarket_data["City"].value_counts().reset_index()
city_counts.columns = ["City", "Counts"]

# Create a donut chart for country
plt.figure(figsize=(8, 8))
plt.pie(
    country_counts["Counts"],
    labels=country_counts["Country"],
    autopct="%1.1f%%",
    startangle=140,
)
plt.axis("equal")  # Equal aspect ratio ensures pie is drawn as a circle.

# Draw a circle at the center of pie to make it a donut
centre_circle = plt.Circle((0, 0), 0.70, fc="white")
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title("Donut Chart of Transactions by Country")
plt.show()

# Create a donut chart for state
plt.figure(figsize=(8, 8))
plt.pie(
    state_counts["Counts"],
    labels=state_counts["State"],
    autopct="%1.1f%%",
    startangle=140,
)
plt.axis("equal")  # Equal aspect ratio ensures pie is drawn as a circle.

# Draw a circle at the center of pie to make it a donut
centre_circle = plt.Circle((0, 0), 0.70, fc="white")
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title("Donut Chart of Transactions by State")
plt.show()

# Create a donut chart for city
plt.figure(figsize=(8, 8))
plt.pie(
    city_counts["Counts"],
    labels=city_counts["City"],
    autopct="%1.1f%%",
    startangle=140,
)
plt.axis("equal")  # Equal aspect ratio ensures pie is drawn as a circle.

# Draw a circle at the center of pie to make it a donut
centre_circle = plt.Circle((0, 0), 0.70, fc="white")
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title("Donut Chart of Transactions by City")
plt.show()

# Bubble plot for country
plt.figure(figsize=(10, 6))
plt.scatter(
    country_counts["Country"],
    country_counts.index,
    s=country_counts["Counts"] * 5,
    alpha=0.5,
)
plt.yticks(ticks=country_counts.index, labels=country_counts["Country"])
plt.xlabel("Countries")
plt.ylabel("Counts")
plt.title("Bubble Plot of Transactions by Country")
plt.tight_layout()  # Add this line to ensure the x ticks are far enough away from the y axis
plt.show()

# Bubble plot for state
plt.figure(figsize=(10, 6))
plt.scatter(
    state_counts["State"],
    state_counts.index,
    s=state_counts["Counts"] * 5,
    alpha=0.5,
)
plt.yticks(ticks=state_counts.index, labels=state_counts["State"])
plt.xlabel("States")
plt.ylabel("Counts")
plt.title("Bubble Plot of Transactions by State")
plt.tight_layout()  # Add this line to ensure the x ticks are far enough away from the y axis
plt.show()

# Bubble plot for city
plt.figure(figsize=(10, 6))
plt.scatter(
    city_counts["City"],
    city_counts.index,
    s=city_counts["Counts"] * 5,
    alpha=0.5,
)

plt.yticks(ticks=city_counts.index, labels=city_counts["City"])
plt.xlabel("Cities")
plt.ylabel("Counts")
plt.title("Bubble Plot of Transactions by City")
plt.tight_layout()  # Add this line to ensure the x ticks are far enough away from the y axis
plt.show()

# %%
# What do you conclude about the relationship of Country with shopping at these stores?
# Count purchases for each department
dept_counts = supermarket_data["Dept"].value_counts().reset_index()
dept_counts.columns = ["Dept", "Purchases"]

# Sort departments by number of purchases
dept_counts_sorted = dept_counts.sort_values(by="Purchases", ascending=False)

# Create a vertical bar chart
plt.figure(figsize=(10, 6))
sns.barplot(
    data=dept_counts_sorted, x="Dept", y="Purchases", order=dept_counts_sorted["Dept"]
)
plt.xticks(rotation=45)  # Rotate value labels for readability
plt.xlabel("Department")
plt.ylabel("Number of Purchases")
plt.title("Number of Purchases by Department")
plt.tight_layout()  # Add this line to ensure the x ticks are not overlapping
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.barplot(
    data=dept_counts_sorted, y="Dept", x="Purchases", order=dept_counts_sorted["Dept"]
)
plt.xlabel("Number of Purchases")
plt.ylabel("Department")
plt.title("Number of Purchases by Department (Horizontal)")
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=dept_counts_sorted, y="Dept", x="Purchases", s=100
)  # Use scatter plot to mimic a dot plot
plt.xlabel("Number of Purchases")
plt.ylabel("Department")
plt.title("Cleveland Dot Plot of Purchases by Department")
plt.show()

# %%
# What are the three top categories of purchased products by department in terms of number of purchases?
top_3_depts = dept_counts_sorted["Dept"].head(3).tolist()
top_3_depts
print(f"The top three departments by number of purchases are: {top_3_depts}")

# %%
# 2b. Revenue Analysis by Country
# Calculate total revenue by country
total_revenue_country = (
    supermarket_data.groupby("Country")["Revenue"].sum().reset_index()
)

# Create a bar chart
plt.figure(figsize=(10, 6))
sns.barplot(data=total_revenue_country, x="Country", y="Revenue")
plt.xlabel("Country")
plt.ylabel("Total Revenue")
plt.title("Total Revenue by Country")
plt.show()

# %%
# Calculate mean revenue by country
mean_revenue_country = (
    supermarket_data.groupby("Country")["Revenue"].mean().reset_index()
)

# Create a bar chart
plt.figure(figsize=(10, 6))
sns.barplot(data=mean_revenue_country, x="Country", y="Revenue")
plt.xlabel("Country")
plt.ylabel("Mean Revenue")
plt.title("Mean Revenue by Country")
plt.show()

# %%
# Calculate overall mean revenue
overall_mean_revenue = supermarket_data["Revenue"].mean()

# Calculate deviation from the mean for each country
mean_revenue_country["Deviation"] = (
    mean_revenue_country["Revenue"] - overall_mean_revenue
)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=mean_revenue_country, x="Country", y="Deviation")
plt.xlabel("Country")
plt.ylabel("Deviation from Mean Revenue")
plt.title("Mean Revenue Deviation by Country")
plt.show()

# %%
# What advice to management would you provide given this analysis?
# exploratory analysis of revenue
# Calculate total revenue by department
total_revenue_dept = supermarket_data.groupby("Country")["Revenue"].sum().reset_index()

# Sort departments by total revenue
total_revenue_dept_sorted = total_revenue_dept.sort_values(by="Revenue", ascending=False)

print(f"The top three countries by total revenue are: {total_revenue_dept_sorted['Country'].head(3).tolist()}")

# The top three countries by total revenue are: ['USA', 'Mexico', 'Canada']

# what does mean revenue deviation by country tell us?
# The mean revenue deviation by country tells us how each country's mean revenue compares to the overall mean revenue. A positive deviation indicates that the country's mean revenue is higher than the overall mean, while a negative deviation indicates that the country's mean revenue is lower than the overall mean. This information can help management identify countries where the supermarket is performing better or worse in terms of revenue compared to the overall average.

# Canada is the highest in terms of mean revenue deviation, which means that the supermarket is performing better in Canada compared to the overall average. On the other hand, Mexico is in the positive but barely and the USA is in the negative. This information can help management identify areas for improvement and focus on strategies to increase revenue in countries where the supermarket is underperforming.

# Based on this we can advise management to focus on strategies to increase revenue in the USA and Mexico, while maintaining the supermarket's performance in Canada and potentially expanding operations in this country.


# %%
# Create a pivot table for total revenue by country
pivot_revenue_country = supermarket_data.pivot_table(
    values="Revenue", index="Country", aggfunc="sum"
).reset_index()

# Display the pivot table
print(pivot_revenue_country)

# %%
# Create a bar chart directly from the pivot table
plt.figure(figsize=(10, 6))
sns.barplot(data=pivot_revenue_country, x="Country", y="Revenue")
plt.xlabel("Country")
plt.ylabel("Total Revenue")
plt.title("Total Revenue by Country (From Pivot Table)")
plt.show()

# %%
# Survey Data
# Load the data
survey_data = pd.read_csv("http://web.pdx.edu/~gerbing/521/resources/460S14.csv")

# Display the first few rows to understand the structure
print(survey_data.head())

# Check for missing data
print(survey_data.isnull().sum())

# Sample size (number of respondents)
print(f"Sample size: {len(survey_data)}")

# %%
response_labels = {1: "Not at all", 2: "Some", 3: "A fair amount", 4: "Cannot remember"}

columns_to_convert = ["Past_1", "Past_2", "Past_3", "Past_4"]
for col in columns_to_convert:
    survey_data[col] = survey_data[col].map(response_labels)

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming survey_data is your DataFrame
plt.figure(figsize=(10, 6))
sns.countplot(x="Past_1", data=survey_data, order=list(response_labels.values()))
plt.xlabel("Past_1 Responses")
plt.ylabel("Count")
plt.title("Distribution of Responses for Past_1")
plt.xticks(rotation=45)
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.countplot(x="Past_1", data=survey_data, order=list(response_labels.values()))
plt.xlabel("Past_1 Responses")
plt.ylabel("Count")
plt.title("Distribution of Responses for Past_1")
plt.xticks(rotation=45)
plt.show()

# %%
ordered_labels = ["Cannot remember", "Not at all", "Some", "A fair amount"]
survey_data["Past_1_ordered"] = pd.Categorical(
    survey_data["Past_1"], categories=ordered_labels, ordered=True
)

# Then, to visualize:
plt.figure(figsize=(10, 6))
sns.countplot(x="Past_1_ordered", data=survey_data, order=ordered_labels)
plt.xlabel("Past_1 Responses (Ordered)")
plt.ylabel("Count")
plt.title("Ordered Responses for Past_1")
plt.xticks(rotation=45)
plt.show()

# %%
melted_data = pd.melt(
    survey_data,
    value_vars=["Past_1", "Past_2", "Past_3", "Past_4"],
    var_name="Question",
    value_name="Response",
)

plt.figure(figsize=(12, 8))
sns.countplot(x="Response", hue="Question", data=melted_data, order=ordered_labels)
plt.xlabel("Responses")
plt.ylabel("Count")
plt.title("Responses for All Past Questions")
plt.xticks(rotation=45)
plt.legend(title="Question")
plt.show()

# %%
import pandas as pd

# Assuming the existence of a 'supermarket_data' DataFrame
# Replace 'TransactionID' with a column that uniquely identifies each transaction if named differently
transactions_by_country_gender = supermarket_data.groupby(
    ["Country", "Gender"], as_index=False
)["Transaction"].count()
transactions_by_country_gender.columns = ["Country", "Gender", "Transactions"]

# %%
import matplotlib.pyplot as plt

pivot_data = transactions_by_country_gender.pivot(
    index="Country", columns="Gender", values="Transactions"
)
pivot_data.plot(kind="bar", stacked=True)
plt.ylabel("Total Transactions")
plt.title("Transactions by Country and Gender (Stacked)")
plt.show()

# %%
pivot_data.plot(kind="barh", stacked=True)
plt.xlabel("Total Transactions")
plt.title("Transactions by Country and Gender (Horizontal Stacked)")
plt.show()

# %%
pivot_data["Total"] = pivot_data.sum(axis=1)
pivot_data_sorted = pivot_data.sort_values(by="Total", ascending=False)
pivot_data_sorted.drop(columns="Total", inplace=True)

pivot_data_sorted.plot(kind="bar", stacked=True)
plt.ylabel("Total Transactions")
plt.title("Transactions by Country and Gender (Stacked, Sorted)")
plt.show()

# %%
sns.barplot(x='Country', y='Transactions', hue='Gender', data=transactions_by_country_gender)
plt.title('Transactions by Country and Gender (Grouped)')
plt.show()
# %%
pivot_data_normalized = pivot_data.div(pivot_data.sum(axis=1), axis=0)
pivot_data_normalized.plot(kind="bar", stacked=True)
plt.ylabel("Percentage of Transactions")
plt.title("Transactions by Country and Gender (100% Stacked)")
plt.show()

# %%
# Conclusion?
# The stacked bar chart shows the total number of transactions
# by country

# The horizontal stacked bar chart provides a different perspective
# on the same data, making it easier to compare the number of transactions
# between countries

# The sorted stacked bar chart helps identify the countries with the
# highest number of transactions and the distribution of
# transactions

# The grouped bar chart provides a clear comparison of the number of
# transactions

# The 100% stacked bar chart shows the percentage of transactions
# by country

# Each of these visualizations provides a different view of the data
# and can be used to answer different questions or highlight different
# aspects of the data. Depending on the specific question or
# analysis being conducted, different visualizations may be more
# appropriate or useful.

# Which form of the bar chart is most useful for comparing gender across countries? Why?
# The grouped bar chart is most useful
# for comparing
# Why?
# The grouped bar chart provides a clear comparison of the number of transactions
