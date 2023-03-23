#!/usr/bin/env python
# coding: utf-8

# First of all I imported the libraries that are essential for our data analysis process.
# We will do make our data clean in order to perform analysis.

# In[6]:


import pandas as pd
import numpy as nm
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.stats.multicomp as mc

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# I installed opendatastes to import data from the website of Kaggle.

# In[7]:


get_ipython().system('pip install opendatasets')


# In[8]:


import opendatasets as od


# In[9]:


dataset = 'https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata'


# In[10]:


od.download(dataset)


# In[11]:


import os


# In[12]:


data_dir = './airbnbopendata'


# In[13]:


os.listdir(data_dir)


# In[14]:


frame = pd.read_csv('Airbnb_Open_Data.csv',low_memory=False)
frame


# Removing the duplicates

# In[15]:


# Count the number of rows in the original DataFrame
print('Number of rows before removing duplicates:', len(frame))

# Drop duplicate rows based on all columns
frame = frame.drop_duplicates()

# Count the number of rows in the updated DataFrame
print('Number of rows after removing duplicates:', len(frame))


# checking that do we have null values in our data.

# In[128]:


print(frame.isnull().sum())


# I wanted to see the column names so, I performed the below code

# In[17]:


print(frame.columns)


# In[18]:


print(frame.head())


# Checking the Shape of the data.

# In[20]:


print(frame.shape)


# Used the mode function to fill the null values of each data. 

# In[21]:


frame_copy=frame.copy()
frame_copy['long'].fillna(frame_copy['long'].mode()[0], inplace=True)
frame_copy['NAME'].fillna(frame_copy['NAME'].mode()[0], inplace=True)
frame_copy['neighbourhood group'].fillna(frame_copy['neighbourhood group'].mode()[0],inplace = True)
frame_copy['neighbourhood'].fillna(frame_copy['neighbourhood'].mode()[0],inplace = True)
frame_copy['host name'].fillna(frame_copy['host name'].mode()[0],inplace=True)
#replacing null values using mode in name column
frame_copy['lat'].fillna(frame_copy['lat'].mode()[0],inplace=True)
#replacing null values using mode in name column
frame_copy['long'].fillna(frame_copy['long'].mode()[0], inplace=True)
#replacing null values using mode in name column
frame_copy['country'].fillna(frame_copy['country'].mode()[0], inplace=True)
#replacing null values using mode in name column
frame_copy['country code'].fillna(frame_copy['country code'].mode()[0], inplace=True)
#replacing null values using mode in name column
frame_copy['instant_bookable'].fillna(frame_copy['instant_bookable'].mode()[0], inplace=True)
#replacing null values using mode in name column
frame_copy['cancellation_policy'].fillna(frame_copy['cancellation_policy'].mode()[0], inplace=True)
frame_copy['host_identity_verified'].fillna(frame_copy['host_identity_verified'].mode()[0], inplace=True)

frame_copy['Construction year'].fillna(frame_copy['Construction year'].mode()[0], inplace=True)
frame_copy['service fee'].fillna(frame_copy['service fee'].mode()[0], inplace=True)
frame_copy['minimum nights'].fillna(frame_copy['minimum nights'].mode()[0], inplace=True)
frame_copy['number of reviews'].fillna(frame_copy['number of reviews'].mode()[0], inplace=True)
frame_copy['last review'].fillna(frame_copy['last review'].mode()[0], inplace=True)
frame_copy['reviews per month'].fillna(frame_copy['reviews per month'].mode()[0], inplace=True)
frame_copy['review rate number'].fillna(frame_copy['review rate number'].mode()[0], inplace=True)
frame_copy['calculated host listings count'].fillna(frame_copy['calculated host listings count'].mode()[0], inplace=True)
frame_copy['availability 365'].fillna(frame_copy['availability 365'].mode()[0], inplace=True)
frame_copy['house_rules'].fillna(frame_copy['house_rules'].mode()[0], inplace=True)
frame_copy['price'].fillna(frame_copy['price'].mode()[0], inplace=True)


# In[22]:


frame = frame_copy.copy()


# In[133]:


print(frame.isnull().sum())


# to remove the license column I used the drop.

# In[134]:


frame = frame.drop("license", axis=1)


# In[135]:


print(frame.isnull().sum())


# The 'types_of_rooms' variable contains the percentage of each type of room in the dataset, rounded to one decimal point. 

# In[136]:


types_of_rooms = frame['room type'].value_counts(normalize=True).mul(100).round(1)
types_of_rooms


# The explanation of the code below,
# 
# The first four lines of code convert the price column from object to float data type, by removing commas and dollar signs, and then parsing it as numeric data using pandas.
# The next two lines of code create two groups based on the room type (Entire home/apt and Private room) and extract the price values for each group.
# The stats.ttest_ind() function from the scipy library is then used to perform an independent two-sample t-test between the two groups, assuming unequal variances.
# 
# Finally, the t-statistic and p-value for the t-test are printed using f-strings.
# 
# The purpose of this code is to perform a statistical analysis to test whether there is a significant difference in mean price between the two groups of listings (Entire home/apt and Private room). The t-test is a common method for comparing means of two groups and determining whether the difference between them is statistically significant or not.

# In[23]:


from scipy import stats
frame['price'] = frame['price'].astype(str)
frame['price'] = frame['price'].str.replace(',', '').str.replace('$', '', regex=False)
frame['price'] = frame['price'].astype(float)
frame['price'] = pd.to_numeric(frame['price'])
# Split data into two groups based on room type
group1 = frame[frame['room type'] == 'Entire home/apt']['price']
group2 = frame[frame['room type'] == 'Private room']['price']

# Perform t-test
t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

# Print results
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")


# In the code below, I performed an ANOVA (Analysis of Variance) test to determine if there are significant differences in mean price between three groups of listings, based on their neighbourhood_group feature.

# In[24]:



# create three groups based on neighbourhood_group
group1 = frame[frame['neighbourhood group'] == 'Brooklyn']['price']
group2 = frame[frame['neighbourhood group'] == 'Manhattan']['price']
group3 = frame[frame['neighbourhood group'] == 'Queens']['price']

# perform ANOVA test
f_statistic, p_value = stats.f_oneway(group1, group2, group3)

# print results
print("F-statistic:", f_statistic)
print("P-value:", p_value)


# In[25]:




# perform one-way ANOVA
f_stat, p_val = stats.f_oneway(frame[frame['neighbourhood group'] == 'Brooklyn']['price'],
                               frame[frame['neighbourhood group'] == 'Manhattan']['price'],
                               frame[frame['neighbourhood group'] == 'Queens']['price'],
                               frame[frame['neighbourhood group'] == 'Staten Island']['price'],
                               frame[frame['neighbourhood group'] == 'Bronx']['price'])

# perform Tukey's HSD post-hoc test
m_comp = mc.MultiComparison(frame['price'], frame['neighbourhood group'])
tukey_res = m_comp.tukeyhsd()

print(tukey_res)


# In[178]:


print(frame['neighbourhood group'].unique())


# In[180]:


frame['neighbourhood group'] = frame['neighbourhood group'].replace({'brookln': 'Brooklyn', 'manhatan': 'Manhattan'})


# In[194]:


print(frame['neighbourhood group'].unique())


# In[198]:


print(frame['room type'].unique())


# Now, In the below code data analysis have been performed by using different types of graphs like barpchart, piechart, boxplot, histogram, scatterplot and violinplot. 

# In[200]:



# group data by room type and calculate average price
avg_price = frame.groupby('room type')['price'].mean()

# create barplot
fig, ax = plt.subplots()
ax.bar(avg_price.index, avg_price.values, color=['blue', 'green', 'red', 'orange'], width=0.5)

# add labels and title
ax.set_xlabel('Room Type')
ax.set_ylabel('Average Price')
ax.set_title('Barplot of Average Price by Room Type')

# create legend
colors = {'Private room': 'blue', 'Entire home/apt': 'green', 'Shared room': 'red', 'Hotel room': 'orange'}
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')

# display plot
plt.show()


# In[207]:



import matplotlib.pyplot as plt

# create a list of labels for each neighbourhood group
labels = ['Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bronx']

# create a list of colors for each neighbourhood group
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# filter the DataFrame to include only the neighbourhood group column
ng_counts = frame['neighbourhood group'].value_counts()

# create a pie chart of neighbourhood group distribution
plt.figure(figsize=(8,6))
plt.pie(ng_counts, labels=ng_counts.index, colors=colors, autopct='%1.1f%%')

# add legend outside the chart
plt.legend(labels=labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# set title
plt.title('Pie Chart of Neighbourhood Group Distribution', fontsize=14)

# show the plot
plt.show()





# In[ ]:





# In[195]:


# create a list of labels for each neighbourhood group
labels = ['Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bronx']

# define colors for each box
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# define box properties
boxprops = dict(linestyle='-', linewidth=2, color='black')

# create a boxplot of price by neighbourhood group
plt.figure(figsize=(8,6))
boxplots = plt.boxplot([frame[frame['neighbourhood group']=='Brooklyn']['price'], 
             frame[frame['neighbourhood group']=='Manhattan']['price'], 
             frame[frame['neighbourhood group']=='Queens']['price'], 
             frame[frame['neighbourhood group']=='Staten Island']['price'], 
             frame[frame['neighbourhood group']=='Bronx']['price']], 
              labels=labels, boxprops=boxprops, patch_artist=True)

# set colors for each box
for patch, color in zip(boxplots['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_label('')

# set axis labels and title
plt.xlabel('Neighbourhood Group', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.title('Boxplot of Price by Neighbourhood Group', fontsize=14)

# add legend
plt.legend(handles=boxplots['boxes'], labels=labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# show the plot
plt.show()


# In[137]:



# Create a horizontal bar chart
colors = ['#5DA5DB', '#FAA43C', '#60BD69', '#F17CB1']
plt.figure(figsize=(8, 6))
ax = plt.gca()
ax.barh(types_of_rooms.index, types_of_rooms.values, color=colors)
plt.title('Percentage of Types of Rooms')
plt.xlabel('Percentage')
plt.show()


# In[138]:




plt.figure(figsize=(8, 6))
sns.histplot(data=frame, x='price', kde=True,color='orange')
plt.title('Histogram of Price')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# In[139]:


import matplotlib.pyplot as plt
color = ['purple']
plt.figure(figsize=(8, 6))
plt.scatter(frame['price'], frame['number of reviews'],color=color)
plt.title('Scatter Plot of Price vs. Number of Reviews')
plt.xlabel('price')
plt.ylabel('number of reviews')
plt.show()


# In[215]:


sns.violinplot(x='room type', y='reviews per month', data=frame, palette='Blues')
plt.title('Distribution of Reviews per Month by Room Type')
plt.show()



# In[ ]:





# In[ ]:





# In[ ]:




