# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

##
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

##
#Configure Visualization Defaults
# %matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8

## Read data
df_train = pd.read_csv(r'F:\GitHubData\Titanic\train.csv')
df_test = pd.read_csv(r'F:\GitHubData\Titanic\test.csv')
print(df_train.info())
print(df_train.sample(1).T)

## Summary
print('-'*10)
print(f'Number of Training Examples: {df_train.shape[0]}')
print(f'Number of Test Examples = {df_test.shape[0]}')
print(f'Shape of Training Examples = {df_train.shape}')
print(f'Shape of Test Examples = {df_test.shape}')
print(sorted(df_train.columns.tolist())) # sort() is not inapplicable
print(sorted(df_test.columns.tolist()))
print(df_train.describe(include=[np.number]).T)
print(df_train.describe(include=['O']).T)

## Missing Values
print('-'*10)
print(df_train.isnull().sum())
print(df_train.columns[df_train.isnull().any()])
print('-'*10)
print(df_test.isnull().sum())
print(df_test.columns[df_test.isnull().any()])

# # Title, Family Size




## # Age
# 
ss_age = pd.qcut(df_train['Age'], 10)
ss_survived = df_train['Survived']
df_qAgeSruvived = pd.concat([ss_age, ss_survived], axis=1)
sns.countplot(x='Age', hue='Survived', data=df_qAgeSruvived)
#
plt.xlabel('Age', size=15, labelpad=20)
plt.ylabel('Passenger Count', size=15, labelpad=20)
plt.tick_params(axis='x', labelsize=10)
plt.tick_params(axis='y', labelsize=10)
plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})
plt.title('Survival Counts by {}'.format('Age'), size=15, y=1.05)
plt.show(block=True)

## # Age plot
h = sns.FacetGrid(df_train, row = 'Sex', col = 'Pclass', hue = 'Survived')
h.map(plt.hist, 'Age', alpha = .75)
h.add_legend()
plt.show(block=True)

# # Impute missing age


# DataFrameSelector(num_attribs) cat_attribs
# FeatureUnion







