import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
# %%
"""
Short Answer Questions
What is a run chart?  
Define a stable process.  
What does the run chart of a stable process look like? What is one way to assess stability?  
Distinguish between a run chart and a time series plot.  
What are the two types of visualizations for displaying multiple time series?
How does the concept of a serial number relate to store dates in Excel and R?  
What is a mapping projection? What is its purpose?
What role does a shapefile provide when creating a map on the computer?
What is the role of a simple features data set in computer map making?
What ggplot2 function provides for specifying the projection underlying a specific map?
What is the distinction between a raster image map and a vector image map?
Maps require geocodes. What are they?
What ggplot2 function provides for plotting a simple features data set? How many such function calls are needed to create a specific map?
What is a choropleth map?
Analysis Problems
1. Plot Time Oriented Data: Run Chart
An example is an assembly line process that fills cereal boxes. To calibrate the machine that releases the cereal into the box, record the weight in grams for 50 consecutive cereal boxes. The target value, the stated filled weight in grams on the cereal box, is 350 grams.
The run chart provides insight into the dynamics of an ongoing process. Because of the random variation inherent in every process outcome, the run chart displays these random fluctuations, present even for a stable process. The visualization of the output of a stable process centers on a horizontal line with random variations about that center line with a constant level of variation. One purpose of the run chart is to evaluate the stability of the process. 
Data:  http://lessRstats.com/data/Cereal.xlsx
a. Construct the run chart of the variable Weight.
b. Does the production process appear stable?
2. Plot Time Oriented Data: Time Series
The following data set updates deaths and other statistic due to COVID-19 on a daily basis. 
The data:  https://opendata.ecdc.europa.eu/covid19/casedistribution/csv
This is a 4.1MB file so you may prefer to download the file to your computer instead of reading from the web each time you re-run your analysis. Download with the R download.file() function.
download.file("https://opendata.ecdc.europa.eu/covid19/casedistribution/csv", "covid.csv")
The second parameter value is the destination file, which will be in the current working directory of your R session. Get the location of your current working directory from getwd(). Put a path in front of the file name to direct somewhere else.
As always, examine the data before beginning analysis.   How many rows of data in the full data set?
 From the output of Read(), we will focus on a subset of variables. Read() also provides the column indices to identify the variables. Subset the data table to include only data for USA, Italy, and Brazil. The corresponding two-character country codes, the variable geoId, are US, IT, and BR. Just retain the variables in columns 1, 5, 6, and 8 through 10. 
The output of Read() contains the integers that describe the ordinal position of the variable in the data frame. Entering these integers instead of the variable names is less typing. The .() is a lessR function to simplify input. 
d <- d[.(geoId %in% c("US","BR","IT")), .(1,5,6,8,10)]
c. After the extraction (sub-setting), examine the first several rows, and the total number of rows. Describe.
d. Do a bar chart of the variable geoId to establish how many records (rows of data) are available from each country that are represented in the data set. 
e. Convert the character string date to an R variable type Date.
f. Apply the Base R range() function to variable in the d data frame d$dateRep to verify the range of dates. What are the beginning and ending dates? 
g. Sort the data by dateRep. Before plotting a time series the rows of data must be ordered by date from earliest to latest.
h. Plot the time series of deaths just for the USA with either lessR or ggplot2. First form a separate data frame, sub-setting again, but leave d alone as we will use that later.
i. Interpret the time series.
j. Get a Trellis plot for the time series of death for USA, Italy, and Brazil. Interpret and compare deaths across the three countries.
3. Country Map 
Display a map of Great Britain that includes all cities larger than 500,000 population with the population mapped into the size of the corresponding plotted point.
In addition to lessR, the following package installations are needed to run these analyses. Note that the rnaturalearth package is not on the official CRAN servers.
install.packages("rnaturalearth", repos="http://packages.ropensci.org", type="binary")
install.packages(c("ggrepl", "sf"))
Get the cities data from geonames.org and then unzip the zipped downloaded file.
download.file("http://download.geonames.org/export/dump/cities15000.zip", "cities15000.zip")
unzip("cities15000.zip")
For the cities15000.txt unzipped data file, the relevant variable is country.code. The value is "GB".
For the ne_states() derived data, the relevant variable is country, with value "united kingdom".

Technical note: We do some subsetting of a data frame in this assignment, that is, dropping rows and/or columns. The base R function Extract provides the most general way to accomplish subsetting, here for a data frame named d: 
d[rows, cols]
Here rows and cols are expressions that indicate the rows and columns to extract.  The lessR dot function .() simplifies the process. Learn more from the vignette called Subset a Data Frame, available from entering the following into R: 
browseVignettes("lessR")
Read the data set of the world's 15000 largest cities from Geonames. Display the output of Read() and the first several lines of the data frame.
Simplify by subsetting the data frame to just cities within Great Britain with the following variables: name, longitude, latitude, population. Subset the rows of the data frame just to cities with a population larger than 500,000. 
Read the map data of Great Britain from the natural earth data source into a simple features data frame.
Merge the map data with the city data into a single simple features data frame. Show the first two or three lines of the data frame (any more just wastes space).
Use ggplot2 to plot three separate layers: The Great Britain map, the cities with corresponding sized points, and the city names.

4. Choropleth Map
Plot a map of the states of the USA, with the fill color of the state depending on the number of deaths per million of COVID-19 in 2022. The darker the color, the larger the death rate. 
To create this map, get the mapping data for the USA states from the maps package, get the COVID data from the indicated Excel file, then merge the data files to add the COVID information state-by-state with the simple features mapping information by state.
Illustrate the map() function from package maps to show a map of the states of the USA. [Follow the example toward the end of Sec 8.1.5 to create the mapping data frame. Drop the plot parameter and the fill parameter from the use of map() in that example.]
Convert the map data to a simple features data frame called states. Do not plot the data. Display the first several lines of the transformed data and describe.
Read the COVID-19 data for the USA into a data frame called covid. Show the first several lines of the data.
Data source: http://web.pdx.edu/~gerbing/521/data/covid19USA.xlsx
The states data lists the states in lower case. The covid data file lists the states in title format, that is, uppercase first letter. To merge on the values of a common variable, the state names, set the state names in the covid data frame to lower case. 
Examine the first several lines of the transformed data file to verify the transformation. Did it work? Why?
Add the covid data to the sf states data file. To merge data frames you need to know exactly the contents of each data frame, particularly the variable on which to merge. 

i) Apply head() to each data frame to merge, states and covid.

ii) Merge the data frames. The Data Wrangling example toward the end of Section 8.1.5 uses the function inner_join.sf() from the sf package to do the merge. The difference in this example is that the variables for which to merge the two data sets have different names, ID and State, though identical data values by which to merge. One option is to rename one of the variables to have the same as the other variable name. More generally, merge on the common data values with the two names in the two data frames.
This is not a course in data manipulation. Here is how to merge the data frames states and covid by their respective common variables with the base R function merge().
states = merge(x=states, y=covid, by.x="ID", by.y="State")
List the first several lines of this merged data set. Describe. What information does the data set contain?
Plot the choropleth map of COVID death rates by USA state. The variable the indicates the deaths per million is Deaths1M.

"""

# %%
# What is a run chart?
# A run chart is a graph that displays data in a time sequence. It is used to detect trends, cycles, and shifts in data over time.

# Define a stable process.
# A stable process is a process that is in control and is not subject to random variation.

# What does the run chart of a stable process look like? What is one way to assess stability?
# The run chart of a stable process looks like a horizontal line with random variations about that center line with a constant level of variation. One way to assess stability is to look for patterns in the data that indicate a lack of randomness.

# Distinguish between a run chart and a time series plot.
# A run chart is a type of time series plot that displays data in a time sequence. A time series plot is a more general term that can refer to any plot that displays data over time.

# What are the two types of visualizations for displaying multiple time series?
# The two types of visualizations for displaying multiple time series are line plots and stacked area plots.

# How does the concept of a serial number relate to store dates in Excel and R?
# A serial number is a unique identifier for a date in Excel and R. It is used to represent dates as numbers, which can be used in calculations and comparisons.

# What is a mapping projection? What is its purpose?
# A mapping projection is a method used to represent the curved surface of the Earth on a flat map. Its purpose is to preserve the relative size, shape, and distance of geographic features.

# What role does a shapefile provide when creating a map on the computer?
# A shapefile provides the geographic data needed to create a map on the computer. It contains information about the shapes, locations, and attributes of geographic features.

# What is the role of a simple features data set in computer map making?
# A simple features data set is a standardized format for representing geographic data in a computer-readable form. It provides a way to store and manipulate geographic data for mapping purposes.

# What ggplot2 function provides for specifying the projection underlying a specific map?
# The ggplot2 function `coord_sf()` provides for specifying the projection underlying a specific map.

# What is the distinction between a raster image map and a vector image map?
# A raster image map is made up of pixels and is best suited for displaying continuous data, while a vector image map is made up of geometric shapes and is best suited for displaying discrete data.

# Maps require geocodes. What are they?
# Geocodes are geographic coordinates that represent a specific location on the Earth's surface. They are used to identify and locate geographic features on a map.

# What ggplot2 function provides for plotting a simple features data set? How many such function calls are needed to create a specific map?
# The ggplot2 function `geom_sf()` provides for plotting a simple features data set. Typically, one function call is needed to create a specific map, but additional calls may be needed to customize the appearance of the map.

# What is a choropleth map?
# A choropleth map is a type of thematic map that uses color shading to represent the spatial distribution of a variable across geographic areas.

# %%
# 1. Plot Time Oriented Data: Run Chart
# Data: http://lessRstats.com/data/Cereal.xlsx
# a. Construct the run chart of the variable Weight.
cereal_data = pd.read_excel("http://lessRstats.com/data/Cereal.xlsx")
plt.plot(cereal_data['Weight'])
plt.title('Run Chart of Weight')
plt.xlabel('Box Number')
plt.ylabel('Weight (grams)')
plt.show()

# b. Does the production process appear stable?
# The production process appears to be stable as there are random fluctuations around a central line with a constant level of variation.

# %%
# 2. Plot Time Oriented Data: Time Series
# The following data set updates deaths and other statistic due to COVID-19 on a daily basis.

# Download the data
covid_data = pd.read_csv("https://opendata.ecdc.europa.eu/covid19/casedistribution/csv")
covid_data.to_csv("covid.csv", index=False)

# Subset the data to include only data for USA, Italy, and Brazil
covid_data_subset = covid_data[covid_data['geoId'].isin(['US', 'IT', 'BR'])][['dateRep', 'geoId', 'deaths']]
print(covid_data_subset.head())
print(covid_data_subset.shape)

# Do a bar chart of the variable geoId
sns.countplot(x='geoId', data=covid_data_subset)
plt.title('Number of Records by Country')
plt.xlabel('Country')
plt.ylabel('Number of Records')
plt.show()

# Convert the character string date to an R variable type Date
covid_data_subset['dateRep'] = pd.to_datetime(covid_data_subset['dateRep'], format='%d/%m/%Y')

# Apply the Base R range() function to variable in the d data frame d$dateRep
print(covid_data_subset['dateRep'].min())
print(covid_data_subset['dateRep'].max())

# Sort the data by dateRep
covid_data_subset = covid_data_subset.sort_values('dateRep')

# Plot the time series of deaths just for the USA
covid_data_usa = covid_data_subset[covid_data_subset['geoId'] == 'US']
plt.plot(covid_data_usa['dateRep'], covid_data_usa['deaths'])
plt.title('Time Series of Deaths in USA')
plt.xlabel('Date')
plt.ylabel('Deaths')
plt.show()

# Get a Trellis plot for the time series of death for USA, Italy, and Brazil
g = sns.FacetGrid(covid_data_subset, col='geoId', col_wrap=3)
g.map(plt.plot, 'dateRep', 'deaths')
plt.show()

# %%
# 3. Country Map
# Display a map of Great Britain that includes all cities larger than 500,000 population with the population mapped into the size of the corresponding plotted point.
import geopandas as gpd
import requests
import zipfile
import io
import pandas as pd

import requests
import zipfile
import io
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

url = "http://download.geonames.org/export/dump/cities15000.zip"

# Download the zip file
response = requests.get(url)
zip_file = zipfile.ZipFile(io.BytesIO(response.content))

# Extract the txt file from the zip file
zip_file.extractall()

# Now you can read the txt file
cities = pd.read_csv("cities15000.txt", sep="\t", header=None, encoding='utf-8')

# Filter cities with population larger than 500,000
cities = cities[cities[14] > 500000]

# Subset the data to just cities within Great Britain with the following variables: name, longitude, latitude, population
cities_gb = cities[cities[8] == 'GB'][[1, 4, 5, 14]]

# Filter the world data to just Great Britain
world_gb = world[world['name'] == 'United Kingdom']

# Merge the map data with the city data into a single simple features data frame
cities_gdf = gpd.GeoDataFrame(cities_gb, geometry=gpd.points_from_xy(cities_gb[5], cities_gb[4]))

# Plot the map
fig, ax = plt.subplots(figsize=(10, 10))
world_gb.boundary.plot(ax=ax)
cities_gdf.plot(ax=ax, markersize=cities_gdf[14] / 100000, color='red', alpha=0.5)
for x, y, label in zip(cities_gdf.geometry.x, cities_gdf.geometry.y, cities_gdf[1]):
    ax.text(x, y, label, fontsize=8)
plt.show()

# %%
# Choropleth Map
# Plot a map of the states of the USA, with the fill color of the state depending on the number of deaths per million of COVID-19 in 2022.
# Data source: http://web.pdx.edu/~gerbing/521/data/covid19USA.xlsx
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# download the data
covid_data = pd.read_excel("http://web.pdx.edu/~gerbing/521/data/covid19USA.xlsx")

# Replace NaN values in 'CasesNow' and 'New_Cases' columns with 0
covid_data['CasesNow'].fillna(0, inplace=True)
covid_data['New_Cases'].fillna(0, inplace=True)

# Read the mapping data for the USA states
usa_states = gpd.read_file('https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip')

# Convert the state names in both dataframes to lowercase
covid_data['State'] = covid_data['State'].str.lower()
usa_states['NAME'] = usa_states['NAME'].str.lower()

# Merge the usa_states and covid_data DataFrames on the state names
merged = usa_states.set_index('NAME').join(covid_data.set_index('State'))

# Replace NaN values in 'Deaths1M' column with 0
merged['Deaths1M'].fillna(0, inplace=True)

# Plot a choropleth map of COVID-19 deaths per million by state
fig, ax = plt.subplots(1, 1)
merged.plot(column='Deaths1M', cmap='YlOrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
plt.show()
# %%
