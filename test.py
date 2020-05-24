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
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8
pd.set_option('display.max_columns', None)

df_train = pd.read_csv(r'F:\GitHubData\Titanic\train.csv')
df_test = pd.read_csv(r'F:\GitHubData\Titanic\test.csv')
df_all = pd.concat([df_train, df_test], join='outer', axis=0)
comb = [df_train, df_test]
df_train.name = 'Training data'
df_test.name = 'Test data'
print(df_train.info())
print(df_train.sample(1).T)

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
        print(f'{col} column missing values (%): {df[col].isnull().sum()/df[col].count():3.2}')
for df in comb:
    print(f'{df.name}')
    display_missing(df)

#Extract Cabin Numbers
df_train['Deck'] = df_train['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
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

#survival rate varies across ticket number prefix
gtb1 = df_train[['Survived','Ticket_alp']].groupby(['Ticket_alp'])
gtb1['Survived'].sum()/gtb1['Survived'].count()

#Ticket Number Distribution by Pclass and Embarked
g = sns.FacetGrid(df_train, col='Pclass', row='Embarked')
g = g.map(sns.countplot, 'Ticket_num')
plt.show()

##
#Only two missing Embarked values. Both are female from class 1 and share the same ticket number.
print(df_train.loc[df_train['Embarked'].isnull(),])

## qcut or check the price value to see whether the same pricing only occurs in a certain port.


##
g = sns.FacetGrid(df_train, col='Deck', row='Pclass')
g = g.map(sns.barplot, y='Survived', x='Embarked')
plt.show()
##Fare Distribution
g = sns.FacetGrid(df_train, row='Pclass', col='Embarked')
g = g.map(plt.hist, 'Fare')
plt.show()


## Age
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








