import seaborn as sns
import pandas as pd

# Import data from shared.py
from shared import df
from shiny.express import input, render, ui

df = pd.read_csv("/Users/hogan/data_mining_hw/BTA-521-data_viz/Employee.csv")

# Page title (with some additional top padding)
ui.page_opts(title=ui.h2("Basic Shiny app", class_="pt-5"))

# Render a histogram of the selected variable (input.var())
@render.plot
def hist():
    p = sns.histplot(df, x=input.var(), facecolor="#007bc2", edgecolor="white")
    return p.set(xlabel=None)

# Convert DataFrame columns to a dictionary
column_dict = df.columns.to_series().to_dict()

# remove Name column
column_dict.pop('Name')

# Select input for choosing the variable to plot using the dictionary
ui.input_select("var", "Variable", column_dict)
