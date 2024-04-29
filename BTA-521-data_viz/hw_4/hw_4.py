# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
# %%
"""
Short Answer Questions
Customize Visualizations
What is the distinction between a temporary and permanent modification of a visual aesthetic for data visualization?
What is the scope of a data visualization? What are the lessR and ggplot2 functions for theme modification?
What is the RGB color space? Why is it the fundamental color space in data visualizations displayed on the computer screen? 
List and describe the dimensions the HCL color space. 
What is the advantage of the HCL color space for data visualizations?
The HCL color space does not have a one-to-one correspondence with the RGB color space. What is the subsequent implication for color display on a computer monitor?
What is a qualitative palette? What is its primary use?
What is a sequential palette? What is its primary use?
What is a divergent palette? What is its primary use?
What is the distinction between changing a theme and changing individual visual aesthetics?
Interactive Visualizations
What is the purpose and goal of an interactive visualization?
What are the two files to create a Shiny visualization? What are their purposes? On what computers do the two files execute?
What is a render function and what is its purpose?
What is an input widget? Illustrate a specific example by discussing how the sliderInput() function works.
What is the advantage of publishing a shiny app on the web?
Go to RStudio.com/download. What are the two options for setting up your own RStudio server? What is the advantage/disadvantage of either?
Worked Problems
Customize Visualizations
1. Color Palettes
Use lessR function getColors() or ggplot2 function show_col() to generate these palettes. Do not need to do both. Any questions about parameters, etc., go to the corresponding manual for the chosen function, ?getColors or ?show_col, as well as the course book. (Accessing the manual for a function should be a common experience of an R user.)
Note also that color scales are color scales regardless of which function generated them. So, getColors() scales can be input into ggplot2 visualization functions and vice versa. 
To read lessR data sets into data frame d:  d <- Read("file reference_of_data_set")
a.	Generate and show HCL qualitative palettes of …
i. Pastel colors
ii. Medium saturation colors
iii. Dark, rich colors.
b. Generate custom fill colors for …
i. For the variable Dept in the Employee lessR data set, do a bar chart with a HCL qualitative palette of dark, rich colors from a.iii above.
ii. For the variable Salary in the Employee lessR data set, do a histogram with a sequential palette of any chosen hue.
iii. For the variable m07 in the Mach4 lessR data set, do a bar chart with a divergent scale with the two hues of your choice. 
The item is "There is no excuse for lying to someone else", assessed with the following set of potential scale responses.
0: Strongly Disagree
1: Disagree
2: Slightly Disagree
3: Slightly Agree
4: Agree
5: Strongly Agree
More properly we would create the factor version of m07 with value labels, and assign a variable label to m07, to make the visualization more informative but need not pursue here.
For the variable Dept in the Employee lessR data set, plot the bar chart with two different themes …
i. theme of your choice
ii. gray scale
Show a color wheel with 50 colors on the wheel and no border between slices, for …
i. A qualitative scale
ii. A sequential scale, hue of your choice
2. Individual Customizations
a.	For either lessR or ggplot2, show the entire list of potential customizations.
b.	For the variable Dept in the Employee lessR data set, customize the bar chart with 
darkred bars, black bar edges, aliceblue panel fill color, darkred variable labels, and slategray axis value labels.
c.	For the variable Price in the lessR StockPrice data set, d <- Read(“StockPrice”), do a time series Trellis plot of Price by Company with a black background and a custom color for the time series curve with a mostly transparent fill of the area under the curve. Colors of your choice.
Interactive Visualizations
Choose any data set with an appropriate continuous variable from which to generate a histogram using either lessR or ggplot2. 
Create a Shiny interactive visualization of a histogram with the following inputs:
i. bin width
ii. bin start
iii. density display
iv. fill bar color 
v. edge bar color
Include 
i. a screen pic of your working Shiny app 
ii. the URL of your Shiny visualization on the web
If you need to know any of the corresponding parameter names, access the manual by entering: ?Histogram. 
View an illustration of all R named colors with lessR showColors(). 
View an illustration of all lessR named sequential scales with lessR showPalettes().
Turn in the URL of the Shiny visualization.
Note: Shiny now also works in Python.

"""

# %%
# Short Answer Questions
# Customize Visualizations
# Q: What is the distinction between a temporary and permanent modification of a visual aesthetic for data visualization?
# A: A temporary modification is a change that is only applied to the current plot, while a permanent modification is a change that is applied to all subsequent plots.
# Q: What is the scope of a data visualization? What are the lessR and ggplot2 functions for theme modification?
# A: The scope of a data visualization is the range of data that is displayed in the plot. The lessR and ggplot2 functions for theme modification are theme() and theme_set() respectively.
# Q: What is the RGB color space? Why is it the fundamental color space in data visualizations displayed on the computer screen?
# A: The RGB color space is a color model that uses red, green, and blue to create colors. It is the fundamental color space in data visualizations displayed on the computer screen because computer monitors use RGB pixels to display colors.
# Q: List and describe the dimensions the HCL color space.
# A: The HCL color space has three dimensions: hue, chroma, and luminance. Hue is the color of the light, chroma is the intensity of the color, and luminance is the brightness of the color.
# Q: What is the advantage of the HCL color space for data visualizations?
# A: The advantage of the HCL color space for data visualizations is that it allows for more accurate and consistent color perception across different devices and lighting conditions.
# Q: The HCL color space does not have a one-to-one correspondence with the RGB color space. What is the subsequent implication for color display on a computer monitor?
# A: The subsequent implication for color display on a computer monitor is that the colors may appear differently on different monitors or devices due to variations in color rendering.
# Q: What is a qualitative palette? What is its primary use?
# A: A qualitative palette is a color palette that is used to distinguish between different categories or groups in a plot. Its primary use is to provide visual differentiation between discrete data points.
# Q: What is a sequential palette? What is its primary use?
# A: A sequential palette is a color palette that is used to represent ordered or continuous data in a plot. Its primary use is to show variations in magnitude or intensity.
# Q: What is a divergent palette? What is its primary use?
# A: A divergent palette is a color palette that is used to represent data with a central value or reference point. Its primary use is to highlight deviations from the central value.
# Q: What is the distinction between changing a theme and changing individual visual aesthetics?
# A: Changing a theme involves modifying the overall appearance of a plot, such as the background color, font size, and grid lines. Changing individual visual aesthetics involves modifying specific elements of a plot, such as the color, size, and shape of data points.

# Interactive Visualizations
# Q: What is the purpose and goal of an interactive visualization?
# A: The purpose and goal of an interactive visualization is to engage users in exploring and interacting with data, allowing them to gain insights and make discoveries through dynamic visualizations.
# Q: What are the two files to create a Shiny visualization? What are their purposes? On what computers do the two files execute?
# A: The two files to create a Shiny visualization are the ui.R and server.R files. The ui.R file defines the user interface layout and controls, while the server.R file contains the server-side logic and data processing. Both files execute on the server where the Shiny application is deployed.
# Q: What is a render function and what is its purpose?
# A: A render function is used in Shiny applications to dynamically update the output based on user inputs or changes in data. It renders the output based on the reactive expressions defined in the server logic.
# Q: What is an input widget? Illustrate a specific example by discussing how the sliderInput() function works.
# A: An input widget is a user interface element that allows users to interact with the Shiny application by providing input values or parameters. The sliderInput() function creates a slider widget that allows users to select a numeric value within a specified range.
# Q: What is the advantage of publishing a shiny app on the web?
# A: The advantage of publishing a Shiny app on the web is that it allows users to access and interact with the application from any device with an internet connection, enabling widespread sharing and collaboration.
# Q: Go to RStudio.com/download. What are the two options for setting up your own RStudio server? What is the advantage/disadvantage of either?
# A: The two options for setting up your own RStudio server are RStudio Server Open Source and RStudio Server Pro. The advantage of RStudio Server Open Source is that it is free and open-source, while the advantage of RStudio Server Pro is that it offers additional features and support for enterprise use.

# Worked Problems (recall are going to be using Python alternatives to R functions)
# Customize Visualizations
# 1. Color Palettes
# a. Generate and show HCL qualitative palettes of …
# i. Pastel colors
# ii. Medium saturation colors
# iii. Dark, rich colors.
# b. Generate custom fill colors for …
# i. For the variable Dept in the Employee lessR data set, do a bar chart with a HCL qualitative palette of dark, rich colors from a.iii above.
# ii. For the variable Salary in the Employee lessR data set, do a histogram with a sequential palette of any chosen hue.
# iii. For the variable m07 in the Mach4 lessR data set, do a bar chart with a divergent scale with the two hues of your choice.
# The item is "There is no excuse for lying to someone else", assessed with the following set of potential scale responses.
# 0: Strongly Disagree
# 1: Disagree
# 2: Slightly Disagree
# 3: Slightly Agree
# 4: Agree
# 5: Strongly Agree
# More properly we would create the factor version of m07 with value labels, and assign a variable label to m07, to make the visualization more informative but need not pursue here.
# For the variable Dept in the Employee lessR data set, plot the bar chart with two different themes …
# i. theme of your choice
# ii. gray scale
# Show a color wheel with 50 colors on the wheel and no border between slices, for …
# i. A qualitative scale
# ii. A sequential scale, hue of your choice
# 2. Individual Customizations
# a. For either lessR or ggplot2, show the entire list of potential customizations.
# b. For the variable Dept in the Employee lessR data set, customize the bar chart with
# darkred bars, black bar edges, aliceblue panel fill color, darkred variable labels, and slategray axis value labels.
# c. For the variable Price in the lessR StockPrice data set, d <- Read(“StockPrice”), do a time series Trellis plot of Price by Company with a black background and a custom color for the time series curve with a mostly transparent fill of the area under the curve. Colors of your choice.
# Interactive Visualizations
# Choose any data set with an appropriate continuous variable from which to generate a histogram using either lessR or ggplot2.
# Create a Shiny interactive visualization of a histogram with the following inputs:
# i. bin width
# ii. bin start
# iii. density display
# iv. fill bar color
# v. edge bar color
# Include
# i. a screen pic of your working Shiny app
# ii. the URL of your Shiny visualization on the web
# If you need to know any of the corresponding parameter names, access the manual by entering: ?Histogram.
# View an illustration of all R named colors with lessR showColors().
# View an illustration of all lessR named sequential scales with lessR showPalettes().
# Turn in the URL of the Shiny visualization.
# Note: Shiny now also works in Python.

# %%
# Load the Employee dataset
employee = pd.read_csv('Employee.csv')

# %%
# Display the first few rows of the dataset
employee.head()

# print the columns
employee.columns

cols_list = [
    "Name",
    "Years",
    "Gender",
    "Dept",
    "Salary",
    "JobSat",
    "Plan",
    "Pre",
    "Post",
]

# %%
# Generate and show HCL qualitative palettes of pastel colors
pastel_colors = sns.color_palette("pastel", as_cmap=True)

# Generate and show HCL qualitative palettes of medium saturation colors
medium_saturation_colors = sns.color_palette("muted", as_cmap=True)

# Generate and show HCL qualitative palettes of dark, rich colors
dark_rich_colors = sns.color_palette("dark", as_cmap=True)

# %%
# Generate custom fill colors for the variable Dept in the Employee dataset
# Bar chart with a HCL qualitative palette of dark, rich colors
plt.figure(figsize=(10, 6))
sns.countplot(x='Dept', data=employee, palette=dark_rich_colors)
plt.title('Bar Chart of Department with Dark Rich Colors')
plt.show()

# Generate custom fill colors for the variable Salary in the Employee dataset
# Histogram with a sequential palette of any chosen hue
plt.figure(figsize=(10, 6))
sns.histplot(x='Salary', data=employee, hue='Dept', palette='viridis')
plt.title('Histogram of Salary with Sequential Palette')
plt.show()

# %%
# Generate custom fill colors for the variable m07 in the Mach4 dataset
# Bar chart with a divergent scale with the two hues of your choice
# http://lessRstats.com/data/name
mach4 = pd.read_csv("http://lessRstats.com/data/Mach4.csv")

# Bar chart with a divergent scale
plt.figure(figsize=(10, 6))

# Define the hues for the divergent scale
hues = ['red', 'blue']

sns.countplot(x='m07', data=mach4, palette=hues)
plt.title('Bar Chart of m07 with Divergent Scale')

plt.show()

# %%
# For the variable Dept in the Employee dataset, plot the bar chart with two different themes
# i. theme of your choice
plt.figure(figsize=(10, 6))
sns.countplot(x='Dept', data=employee, palette='viridis')
plt.title('Bar Chart of Department with Custom Theme')
plt.show()

# ii. gray scale
plt.figure(figsize=(10, 6))
sns.countplot(x='Dept', data=employee, palette='gray')
plt.title('Bar Chart of Department with Gray Scale Theme')
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Show a color wheel with 50 colors on the wheel and no border between slices
# A qualitative scale
plt.figure(figsize=(10, 6))
sns.palplot(sns.color_palette("tab20", n_colors=50))
plt.title('Color Wheel with 50 Colors - Qualitative Scale')
plt.show()

# A sequential scale, hue of your choice
plt.figure(figsize=(10, 6))
sns.palplot(sns.color_palette("viridis", n_colors=50))
plt.title('Color Wheel with 50 Colors - Sequential Scale')
plt.show()

# %%
# show the entire list of potential customizations
import seaborn as sns

# List all the attributes and methods of the seaborn module
print(dir(sns))
# %%
# Individual Customizations
import seaborn as sns
import matplotlib.pyplot as plt

# Set the style and color of the plot
sns.set_theme(style="whitegrid", rc={"axes.facecolor": "aliceblue", "grid.color": "black", "text.color": "slategray", "axes.labelcolor": "darkred", "xtick.color": "slategray", "ytick.color": "slategray"})

# For the variable Dept in the Employee dataset, customize the bar chart with darkred bars, black bar edges, aliceblue panel fill color, darkred variable labels, and slategray axis value labels
plt.figure(figsize=(10, 6))
sns.countplot(x='Dept', data=employee, palette=['darkred'], edgecolor='black')
plt.title('Bar Chart of Department with Custom Colors')
plt.show()

# %%
# Load the StockPrice dataset
stock_price = pd.read_csv("http://lessRstats.com/data/Stocks.csv")

# %%
# Create a Shiny interactive visualization of a histogram with the following inputs:
# i. bin width
# ii. bin start
# iii. density display
# iv. fill bar color
# v. edge bar color

# Include
# i. a screen pic of your working Shiny app
# ii. the URL of your Shiny visualization on the web
# If you need to know any of the corresponding parameter names, access the manual by entering: ?Histogram.

# View an illustration of all R named colors with lessR showColors().
# View an illustration of all lessR named sequential scales with lessR showPalettes().
# Turn in the URL of the Shiny visualization.
# Note: Shiny now also works in Python.

# %%
# Interactive Visualizations
# Choose any data set with an appropriate continuous variable from which to generate a histogram using either lessR or ggplot2, we will use employee dataset
import pandas as pd
import plotly.express as px

# Load the Employee dataset
employees = pd.read_csv('Employee.csv')

# Generate an interactive histogram of the 'Salary' column
fig = px.histogram(employees, x='Salary')

# Show the plot
fig.show()

# %%
# create a shiny app
# Create a Shiny app in Python
from shiny.express import input, render, ui

# Define the UI layout
# Page title (with some additional top padding)
ui.page_opts(title=ui.h2("Basic Shiny app", class_="pt-5"))


# Render a histogram of the selected variable (input.var())
@render.plot
def hist():
    p = sns.histplot(employees, x=input.var(), facecolor="#007bc2", edgecolor="white")
    return p.set(xlabel=None)

ui.input_select("var", "Select a variable", employees.columns)
# %%
