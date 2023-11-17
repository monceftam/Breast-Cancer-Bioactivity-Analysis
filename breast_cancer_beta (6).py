#!/usr/bin/env python
# coding: utf-8

# In[3]:


#import necessary libraries
import pandas as pd
from chembl_webresource_client.new_client import new_client


# In[4]:


#initialize new client
target = new_client.target


# In[6]:


target_query = target.search("breast cancer")
targets = pd.DataFrame.from_dict(target_query)
# Display information about each target
print("Targets related to 'breast cancer':")
print(targets[['target_chembl_id', 'organism', 'target_type', 'pref_name']])



# In[7]:


selected_target = targets.target_chembl_id[2]


# In[8]:


#get activity
activity = new_client.activity
res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")
df = pd.DataFrame.from_dict(res)
print(df.head(3))


# In[9]:


df.to_csv('breast_cancer_data_raw.csv', index=False)


# In[10]:


df2 = df[df.standard_value.notna()]


# In[11]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df2' is your DataFrame which should be loaded or defined earlier in your script
# For example, df2 = pd.read_csv('your_data_file.csv') or any other way we have

# Convert the 'standard_value' column to numeric, forcing non-numeric to NaN
df2['standard_value'] = pd.to_numeric(df2['standard_value'], errors='coerce')

# Drop rows with NaN 'standard_value'
df2.dropna(subset=['standard_value'], inplace=True)

# Calculate pIC50 values
df2['pIC50'] = -np.log10(df2['standard_value'] * (10**-9))

# Determine cutoffs for 'active' and 'inactive' based on domain knowledge or distribution
# For this example, we use arbitrary cutoffs, but you should use scientifically justified cutoffs
active_cutoff = np.percentile(df2['pIC50'], 75)
inactive_cutoff = np.percentile(df2['pIC50'], 25)

# Categorize the bioactivity based on pIC50 values
def categorize_activity(value, active_threshold, inactive_threshold):
    if value >= active_threshold:
        return "active"
    elif value <= inactive_threshold:
        return "inactive"
    else:
        return "intermediate"

# Apply categorization function to each pIC50 value
df2['bioactivity_class'] = df2['pIC50'].apply(categorize_activity, args=(active_cutoff, inactive_cutoff))

# Visualize the distribution of pIC50 values
sns.histplot(df2['pIC50'], bins=30, kde=False)
plt.xlabel('pIC50')
plt.ylabel('Frequency')
plt.title('Distribution of pIC50 values')
plt.show()

# Print the head of the updated DataFrame
print(df2[['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'pIC50', 'bioactivity_class']].head())



# In[12]:


# Filter for compounds with pIC50 values greater than or equal to 8.5
high_potency_df = df2[df2['pIC50'] >= 8.5]

high_potency_df.to_csv('breast_cancer_high_potency_compounds.csv', index=False)


# In[13]:


sns.histplot(high_potency_df['pIC50'], bins=30, kde=False)
plt.xlabel('pIC50')
plt.ylabel('Frequency')
plt.title('Distribution of High-Potency pIC50 Values')
plt.show()


# In[14]:


# Descriptive statistics for pIC50 values
descriptive_stats = df2['pIC50'].describe()
print(descriptive_stats)


# In[16]:


# Sort the DataFrame by pIC50 in descending order to get the most potent compounds at the top
sorted_high_potency_df = high_potency_df.sort_values(by='pIC50', ascending=False)

# Select the top five most potent compounds
top_five_compounds = sorted_high_potency_df.head(5)

# Print the CHEMBL IDs of the top five most potent compounds
print("Top 5 Potential Molecules:")
print(top_five_compounds[['molecule_chembl_id', 'pIC50']])


# In[ ]:




