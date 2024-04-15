# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so

# %%
# load hw_1/SupermarketTransactions.xlsx
cars93_data = pd.read_csv(
    "cars93.csv"
)
cars93_data.head()

# %%
# get columns as list
columns = cars93_data.columns.tolist()
columns

# %%
"""
columns of the data: ['Make',
 'Model',
 'Type',
 'MinPrice',
 'MidPrice',
 'MaxPrice',
 'MPGcity',
 'MPGhiway',
 'Airbags',
 'DriveTrain',
 'Cylinders',
 'Engine',
 'HP',
 'RPM',
 'RevMile',
 'Manual',
 'FuelCap',
 'PassCap',
 'Length',
 'Wheelbase',
 'Width',
 'Uturn',
 'RearSeat',
 'LugCap',
 'Weight',
 'Source']

1. Factors
a. First task identifies categorical variables in the analysis and convert to factors. The output of Read() is designed to provide this information. From that output, why are Type, Manual, and Cylinders categorical variables?
b. From the following information, convert each of these variables to factors.
Type: type of car with approximate ordering from the smallest car to the van
Manual: Manual transmission available, 0 = No, 1 = Yes
Source: 0=non-USA, 1=USA 
2. Univariate Continuous
a. Provide the histogram using default bins for city MPG. 
    i. lessR. ii. ggplot2
b. Provide the histogram with a more appropriate bin width for the city MPG.
    i. lessR. ii. ggplot2
c. Provide the density curve for city MPG. Display the entire range of the curve.
    i. lessR. ii. ggplot2
d. Provide the overlapping density curves for city MPG for USA and non-USA cars. (either lessR or ggplot2)
e. Provide the box plots that compare city MPG for Type of vehicle. (either lessR or ggplot2)
f. Provide the integrated Violin/Box/Scatterplot (VBS) for city MPG. (either lessR or ggplot2)
g. Provide the integrated Violin/Box/Scatter Trellis plots (VBS) for city MPG that compare city MPG for Type of vehicle. (either lessR or ggplot2)
3. Scatterplot
Examine the relationship between HP and MPG highway. First look at the overall relationship and then look at the relationship across different groups. The last question in the sequence looks at the relationship across three different groups, a total of five variables in the scatterplots.
Provide the scatter plot for MPG highway and HP, with the 95% confidence data ellipse.
i. lessR. ii. ggplot2
[for the rest of the scatterplot questions, use any function you wish]
Provide the Trellis (facet) scatterplots for MPG highway and HP for USA vs non-USA cars.
Provide the scatter plot for MPG highway and HP for USA vs non-USA cars plotted on the same panel, with the corresponding regression lines. Interpret.
Plot the scatterplot for MPG highway and HP, mapping MidPrice to the size of each bubble.
Show and interpret the scatterplot matrix for MidPrice, MPGcity, MPGhiway, HP, and Weight.
Plot the scatterplot of HP to city MPG, with Source on each panel with Trellis plots across Transmn and Type. What do you conclude?
"""

# %%
# 1. Factors
# a. First task identifies categorical variables in the analysis and convert to factors. The output of Read() is designed to provide this information. From that output, why are Type, Manual, and Cylinders categorical variables?
# answer: Type, Manual, and Cylinders are categorical variables because they have a limited number of unique values and they are not continuous variables.
# b. From the following information, convert each of these variables to factors.
# Type: type of car with approximate ordering from the smallest car to the van
# Manual: Manual transmission available, 0 = No, 1 = Yes
# Source: 0=non-USA, 1=USA 
cars93_data['Type'] = pd.Categorical(cars93_data['Type'], ordered=True)
cars93_data['Manual'] = pd.Categorical(cars93_data['Manual'], categories=[0, 1])
cars93_data['Cylinders'] = pd.Categorical(cars93_data['Cylinders'])

# 2. Univariate Continuous
# a. Provide the histogram using default bins for city MPG. 
#     i. lessR. ii. ggplot2
plt.hist(cars93_data['MPGcity'])
plt.show()

# b. Provide the histogram with a more appropriate bin width for the city MPG.
#     i. lessR. ii. ggplot2
plt.hist(cars93_data['MPGcity'], bins=20)
plt.show()

# c. Provide the density curve for city MPG. Display the entire range of the curve.
#     i. lessR. ii. ggplot2
sns.kdeplot(cars93_data['MPGcity'])
plt.show()

# d. Provide the overlapping density curves for city MPG for USA and non-USA cars. (either lessR or ggplot2)
sns.kdeplot(cars93_data[cars93_data['Source'] == 0]['MPGcity'], label='non-USA')
sns.kdeplot(cars93_data[cars93_data['Source'] == 1]['MPGcity'], label='USA')
plt.legend()
plt.show()

# e. Provide the box plots that compare city MPG for Type of vehicle. (either lessR or ggplot2)
sns.boxplot(x=cars93_data['Type'], y=cars93_data['MPGcity'])
plt.show()

# f. Provide the integrated Violin/Box/Scatterplot (VBS) for city MPG. (either lessR or ggplot2)
sns.violinplot(x=cars93_data['MPGcity'])
sns.boxplot(x=cars93_data['MPGcity'], width=0.2, color='white')
sns.stripplot(x=cars93_data['MPGcity'], color='black', alpha=0.5)
plt.show()

# g. Provide the integrated Violin/Box/Scatter Trellis plots (VBS) for city MPG that compare city MPG for Type of vehicle. (either lessR or ggplot2)
sns.violinplot(x=cars93_data['Type'], y=cars93_data['MPGcity'])
sns.boxplot(x=cars93_data['Type'], y=cars93_data['MPGcity'], width=0.2, color='white')
sns.stripplot(x=cars93_data['Type'], y=cars93_data['MPGcity'], color='black', alpha=0.5)
plt.show()

# 3. Scatterplot
# Examine the relationship between HP and MPG highway. First look at the overall relationship and then look at the relationship across different groups. The last question in the sequence looks at the relationship across three different groups, a total of five variables in the scatterplots.
# Provide the scatter plot for MPG highway and HP, with the 95% confidence data ellipse.
# i. lessR. ii. ggplot2
sns.scatterplot(x=cars93_data['HP'], y=cars93_data['MPGhiway'])
plt.show()

# [for the rest of the scatterplot questions, use any function you wish]
# Provide the Trellis (facet) scatterplots for MPG highway and HP for USA vs non-USA cars.
sns.relplot(x='HP', y='MPGhiway', hue='Source', data=cars93_data)
plt.show()

# Provide the scatter plot for MPG highway and HP for USA vs non-USA cars plotted on the same panel, with the corresponding regression lines. Interpret.
sns.lmplot(x='HP', y='MPGhiway', hue='Source', data=cars93_data)
plt.show()

# Plot the scatterplot for MPG highway and HP, mapping MidPrice to the size of each bubble.
sns.scatterplot(x='HP', y='MPGhiway', size='MidPrice', data=cars93_data)
plt.show()

# Show and interpret the scatterplot matrix for MidPrice, MPGcity, MPGhiway, HP, and Weight.
sns.pairplot(cars93_data[['MidPrice', 'MPGcity', 'MPGhiway', 'HP', 'Weight']])
plt.show()

# Plot the scatterplot of HP to city MPG, with Source on each panel with Trellis plots across Manual and Type. What do you conclude?
sns.relplot(x='HP', y='MPGcity', hue='Source', col='Manual', row='Type', data=cars93_data)
plt.show()

# What do you conclude?
# 1. The relationship between HP and MPG highway is negative.
# 2. The relationship between HP and MPG highway is negative for both USA and non-USA cars.
# 3. The relationship between HP and MPG highway is negative for both USA and non-USA cars. The relationship is stronger for non-USA cars.
# 4. The relationship between HP and MPG highway is negative. The size of the bubble is proportional to the MidPrice.
# 5. The scatterplot matrix shows the relationship between MidPrice, MPGcity, MPGhiway, HP, and Weight.
# %%
