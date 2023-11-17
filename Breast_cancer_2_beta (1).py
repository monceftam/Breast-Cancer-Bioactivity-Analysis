#!/usr/bin/env python
# coding: utf-8

# In[6]:


# In[1]: Import necessary libraries
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, PandasTools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


# In[7]:


df = pd.read_csv('breast_cancer_high_potency_compounds.csv')



# In[8]:


#  Define Lipinski's descriptors calculation function
def lipinski(smiles, verbose=False):
    moldata = []
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        if mol is not None:  # Only append if the molecule is valid
            moldata.append(mol)
    descriptors = pd.DataFrame([
        [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol)
        ]
        for mol in moldata
    ], columns=["MW", "LogP", "NumHDonors", "NumHAcceptors"])
    return descriptors


# In[45]:


# Define Lipinski's descriptors calculation function
def lipinski(smiles, verbose=False):
    moldata = [Chem.MolFromSmiles(elem) for elem in smiles]
    descriptors = pd.DataFrame([
        [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol)
        ]
        for mol in moldata
    ], columns=["MW", "LogP", "NumHDonors", "NumHAcceptors"])
    return descriptors


# In[9]:


#  Apply Lipinski's function and normalize values
df_lipinski = lipinski(df['canonical_smiles'])
df_combined = pd.concat([df, df_lipinski], axis=1)

def norm_value(input_df):
    norm = input_df['standard_value'].apply(lambda x: min(x, 100000000))
    input_df['standard_value_norm'] = norm
    return input_df

df_combined = norm_value(df_combined)


# In[10]:


#  Convert IC50 to pIC50
def pIC50(input_df):
    pIC50_values = -np.log10(input_df['standard_value_norm'] * (10**-9))
    input_df['pIC50'] = pIC50_values
    return input_df

df_combined = pIC50(df_combined)


# In[11]:


#  Add molecule column to the DataFrame
PandasTools.AddMoleculeColumnToFrame(df_combined, 'canonical_smiles', 'Molecule')


# In[16]:


#  Define the statistical analysis function
def mannwhitney(descriptor, df_analysis):
    # Compare active vs inactive compounds
    active = df_analysis[df_analysis['bioactivity_class'] == 'active'][descriptor]
    inactive = df_analysis[df_analysis['bioactivity_class'] == 'inactive'][descriptor]
    stat, p = mannwhitneyu(active, inactive)
    # Interpret the results of the test
    interpretation = 'Different distribution (reject H0)' if p < 0.05 else 'Same distribution (fail to reject H0)'
    # Compile the results into a DataFrame
    results = pd.DataFrame({
        'Descriptor': [descriptor],
        'Statistics': [stat],
        'p': [p],
        'alpha': [0.05],
        'Interpretation': [interpretation]
    })
    return results


# In[17]:


#  Save plots and results to files
def save_plots_and_results(df_analysis):
    sns.set(style='ticks')
    descriptors = ['pIC50', 'MW', 'LogP', 'NumHDonors', 'NumHAcceptors']
    for descriptor in descriptors:
        plt.figure(figsize=(5.5, 5.5))
        sns.boxplot(x='bioactivity_class', y=descriptor, data=df_analysis)
        plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
        plt.ylabel(descriptor, fontsize=14, fontweight='bold')
        plt.savefig(f'plot_{descriptor}.pdf')
        results = mannwhitney(descriptor, df_analysis)
        results.to_csv(f'mannwhitney_{descriptor}.csv')




# In[18]:


# Assuming df_combined has a 'bioactivity_class' column
save_plots_and_results(df_combined)


# In[19]:


# In[9]: Visualize Lipinski's descriptors with boxplots
lipinski_descriptors = ['MW', 'LogP', 'NumHDonors', 'NumHAcceptors']

for descriptor in lipinski_descriptors:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='bioactivity_class', y=descriptor, data=df_combined)
    plt.title(f'Boxplot of {descriptor} by Bioactivity Class')
    plt.savefig(f'boxplot_{descriptor}.png')  # Saving the plot as a PNG image
    plt.show()


# In[20]:


# In[10]: Visualize the distribution of pIC50 values
plt.figure(figsize=(10, 6))
sns.histplot(df_combined, x='pIC50', hue='bioactivity_class', element='step', stat='density', common_norm=False)
plt.title('Distribution of pIC50 Values by Bioactivity Class')
plt.xlabel('pIC50')
plt.ylabel('Density')
plt.savefig('distribution_pIC50.png')
plt.show()


# In[21]:


#  Visualize the correlation matrix as a heatmap
correlation_matrix = df_combined[lipinski_descriptors].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Lipinski Descriptors')
plt.savefig('correlation_heatmap.png')
plt.show()


# In[ ]:




