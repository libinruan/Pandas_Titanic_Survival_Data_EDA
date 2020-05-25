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

#Configure Visualization Defaults
mpl.style.use('ggplot')
pylab.rcParams['figure.figsize'] = 12,8
sns.set_style('white')
pd.set_option('display.max_columns', None)

df_train = pd.read_csv(r'F:\GitHubData\Titanic\train.csv')
df_test = pd.read_csv(r'F:\GitHubData\Titanic\test.csv')
df_all = pd.concat([df_train, df_test], join='outer', axis=0)
df_train.name = 'Training data'
df_test.name = 'Test data'
print(df_train.info())
print(df_train.sample(1).T)

# comb = [df_train, df_test]

#Dataset Dimensions
print('-'*30)
print(f'Number of Training Examples: {df_train.shape[0]}')
print(f'Number of Test Examples = {df_test.shape[0]}')
print(f'Shape of Training Examples = {df_train.shape}')
print(f'Shape of Test Examples = {df_test.shape}')

#Column name lists
print(sorted(df_train.columns.tolist())) # sort() is not inapplicable
print(sorted(df_test.columns.tolist()))

#Training data - categorical and numeric columns
print('-'*30)
print(df_train.describe(include=[np.number]).T)
print(df_train.describe(include=['O']).T)

#Stat about missing values
print('-'*30)
def display_missing(df):
    for col in df.columns.tolist():
        print(f'{col} column missing values (%): {df[col].isnull().sum()}')

for df in [df_train, df_test]:
    print(f'{df.name}')
    display_missing(df)

#Extract Cabin Numbers
df_train['Deck'] = df_train['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

##Add percentage bar number 1

#
idx = df_train[df_train['Deck'] == 'T'].index
df_train.loc[idx, 'Deck'] = 'A'

##Add percentage bar number 2

##Family size

#Deck and Embarked combined could be a good predictor
df_train[['Deck','Embarked','Sex','Survived']].groupby(['Sex','Deck','Embarked']).mean()

#Percentage of passenger by Embarked Ports but shown in order of Deck and Embarked
deck_embarked = df_train[['Deck','Embarked','Survived']].groupby(['Deck','Embarked']).count()
#solution 1
tb1 = deck_embarked.groupby(level=1).apply(lambda x: 100 * x / float(x.sum()))
tb1.rename(columns={'Survived':'Passengers (%)'})
#solution 2
tb2 = 100 * deck_embarked / deck_embarked.groupby(level=1).transform('sum')
tb2.rename(columns={'Survived':'Passengers (%)'})

#Survival rate by Embarked Port but shown in order of Deck and Embarked
alive_deck_embarked = df_train[['Deck', 'Embarked', 'Survived']].groupby(['Deck','Embarked'])
tb3 = alive_deck_embarked['Survived'].sum()/alive_deck_embarked['Survived'].count()
tb3.sort_values(ascending=False)

##
def getTicketStrNum(df, col):
    temp = df.copy()
    col_num = col + '_num'
    col_alp = col + '_alp'
    temp[col_num] = temp[col].str.extract(r'(\d+)$') # get the last group of numbers
    temp[col_num].fillna(-1, inplace=True)
    temp[col_alp] = temp[col].str.extract(r'(.*)\ \d+$').replace({'\.':'','/':''},regex=True)
    temp[col_alp].fillna('M', inplace=True)
    return temp
df_train = getTicketStrNum(df_train, 'Ticket')
df_train['Ticket_num'] = pd.to_numeric(df_train['Ticket_num'])

#survival rate varies across ticket number prefix
gtb1 = df_train[['Survived','Ticket_alp']].groupby(['Ticket_alp'])
gtb1['Survived'].sum()/gtb1['Survived'].count()

#Ticket Number Distribution by Pclass and Embarked
#The plot doesn't help to impute the two missing Embarked value, both of which are in Pclass = 1.
#The only information gain is that given they share the same ticket number, they should know each
#other and highly likely embark from either C or S together.
g = sns.FacetGrid(df_train, col='Pclass', row='Embarked')
g = g.map(sns.countplot, 'Ticket_num')
plt.show()

##
g = sns.FacetGrid(df_train, col='Pclass', row='Embarked', hue='Deck')
g = g.map(plt.scatter, 'Age', 'Fare')
g.add_legend()
plt.show()

#Only two missing Embarked values. Both are female from class 1 and share the same ticket number.
#Sort by age
print(df_train.loc[df_train['Embarked'].isnull(),].sort_values(by=['Age'], ascending=False))

##
#They more likely board on the ship at port S -- Theory 1.
df_train.loc[df_train['Ticket_num'].between(100000,125000)]['Embarked'].value_counts() # S
df_train.loc[df_train['Fare'].between(60,100)]['Embarked'].value_counts() # S
df_train['Embarked'] = df_train['Embarked'].fillna('S')

#Impute missing age by sex and class group
df_train['Age'] = df_train.groupby(['Sex','Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

#Distribute of minor; the group with higher young people has higher mortality rate
cols = ['Deck','Pclass']
df_train.groupby(cols).filter(lambda x: x['Age'].quantile(q=0.75) > 50)['Survived'].mean()
df_train.groupby(cols).filter(lambda x: x['Age'].quantile(q=0.75) < 30)['Survived'].mean()


##Plot training set survival distribution
#https://i.postimg.cc/25rVKwxB/1590377048.png
#https://python-graph-gallery.com/13-percent-stacked-barplot/

##Categorical variable plot

##Continuous variable plot

##Fare binning with qcut or cut

