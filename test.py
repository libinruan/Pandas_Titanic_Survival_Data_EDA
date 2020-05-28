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
def displayMissing(df):
    for col in df.columns.tolist():
        print(f'{col} column missing values: {df[col].isnull().sum()}')

for df in [df_train, df_test]:
    print(f'{df.name}')
    displayMissing(df)

#Ticket combination is the feature without any missing values.
#We should try to extract any information from it although it appears useless at the first glance.
def getTicketPrefixAndNumber(df, col):
    # naming the columns to be created
    col_num = col + '_num'
    col_alp = col + '_alp'

    # get the last group of contiguous digits
    df[col_num] = df[col].str.extract(r'(\d+)$')
    df[col_num].fillna(-1, inplace=True)

    # get the entire string before a space followed by the last digit group
    df[col_alp] = df[col].str.extract(r'(.*)\ \d+$').replace({'\.':'','/':''},regex=True)
    df[col_alp].fillna('M', inplace=True)
    return df

getTicketPrefixAndNumber(df_all, 'Ticket')
df_all['Ticket_num'] = pd.to_numeric(df_all['Ticket_num'])

#todo: Ticket number should be treated as a nominal variable, instead of a continuous variable

#check to see if the extraction works as expected
colnames = ['Ticket' + s for s in ['','_num','_alp']]
df_all[colnames]

#survival rate varies across ticket number prefix; it can be a predictor
#Does ticket prefix associate with family name?
gtb1 = df_all.iloc[:df_train.shape[0]][['Survived','Ticket_alp']].groupby(['Ticket_alp'])
temp = (gtb1['Survived'].sum()/gtb1['Survived'].count() * 100).sort_values()
temp.name = 'PrefixSurvival'
df_all = pd.merge(df_all, temp, on = 'Ticket_alp')

##The size of each travelling team
grouped = df_all.groupby(['Ticket_num'])
big_team_list = []
for i, g in grouped.groups.items():
    big_team_list.append((i,len(g)))
temp = pd.DataFrame(np.array(big_team_list), columns=['Ticket_num','TeamSize']).sort_values(by='TeamSize')
df_all = pd.merge(df_all, temp, on = 'Ticket_num')

#So it appears team size is a useful predictor too.
print(df_all.groupby(['TeamSize'])['Survived'].mean())
print(df_all.groupby(['Pclass','TeamSize'])['Survived'].mean())
print(df_all.groupby(['Pclass','TeamSize'])['Survived'].mean().reset_index().sort_values(by='Survived'))

##
#todo: who is a child?
def getRole(df, cutoff=7):
    df['Role'] = 'Man'
    df.loc[df['Sex'] == 'female', 'Role'] = 'Woman'
    df.loc[df['Age'] <= cutoff, 'Role'] = 'Child'
    return df
ans = []
ages = range(1, 30)
for cut in ages:
    getRole(df_all, cutoff=cut)
    g = df_all.groupby(['Role'])
    ans.append((g['Survived'].sum() / g['Survived'].count()).to_frame())
temp = pd.concat(ans, axis=1) # 3 by N (=len(ages))
tb1 = pd.DataFrame(np.array(temp).T, columns=['Child', 'Man', 'Woman'], index=ages) # N by 3
tb1.index.name = 'Age'
tb1.reset_index(inplace=True) # prep for melt
tb2 = pd.melt(tb1, id_vars=['Age'], value_vars=['Child', 'Man', 'Woman'], var_name='Role', value_name='Survival')

#seaborn FacetGrid: [link](https://seaborn.pydata.org/generated/seaborn.FacetGrid.html)
g = sns.FacetGrid(tb2, col='Role', margin_titles=True)
g = g.map(plt.plot, 'Age', 'Survival')
#add vertical line
axes = g.fig.axes
for ax in axes:
    ax.vlines(x=15, ymax=1, ymin=0, linestyles='dashed', alpha=0.3, colors='blue')
plt.show()

#I will temporarily set cutoffChildAge = 6 (70% survival rates) to approximate the survival rate of women
getRole(df_all, cutoff=15)

#ok.above
#todo: does the children travel with adults? are they more likely to survive?

##
#How many children in this group?
#Don't put ['Role'] inside the transform's lambda function
df_all['NumChild'] = df_all.groupby('Ticket_num')['Role'].transform(lambda x: (x=='Child').sum())

#although the survival rate by number of child varies, the highest survival rate
#falls in family of three children and it holds across classes.
print(df_all.groupby(['Pclass','NumChild'])['Survived'].mean())


#todo: is family size more informative than team size? check to see if servants are
#more likely to die.


##
def getCabinPrefix(df):
    # 'M' is assigned to missing values
    df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
    return df

getCabinPrefix(df_all)
df_all['Deck'].unique()

#Check to see if the 2nd half of the combined table are all NaN Survived data
#(1) iloc works with slicing that includes right endpoint.
#(2) iloc works with index only, so even though I need 'Survived, I use it separately.
#(3) isnull() to see if there is any missing value
df_all.iloc[df_train.shape[0]:,]['Survived'].isnull().all()

#Cabin numbers have clusters
df_all['Cabin'].value_counts()
#For example, 'B57 B59 B63 B66' corresponds to five persons
#in the Ryerson family. People in the same cabin share the same
#Ticket_alp and Ticket_num. These three variables should be highly
#correlated.
df_all.loc[df_all['Cabin']=='B57 B59 B63 B66']
#'B57 B59 B63 B66' maps to Ticket_alp = 'PC', which is a much larger group.
df_all.loc[df_all['Ticket_alp']=='PC']['Survived'].sum()
#We may check later whether each group can be identified or associated with higher servival rate
#We may also check to see if couples have higher survival rates
#Check Family Ryerson. The number of SibSp and Parch might have more information.


##
#Extract names and titles
def getLastNameAndTitle(df):
    # (1) https://docs.python.org/2/library/re.html
    # (2) Why this patterns works? See the [reason](https://stackoverflow.com/questions/12148784/extract-text-before-first-comma-with-regex#answer-12148814).
    # (3) This pattern works as well r'^([^,]*)'  See the reference [link](https://stackoverflow.com/questions/12187287/capturing-string-right-before-comma-in-regex/12187415#answer-12187415)
    df['LastName'] = df['Name'].str.extract(r'^(.+?),')
    df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
    return df

getLastNameAndTitle(df_all)
cols = ['Name','Title','LastName']
#df_all[cols] works as well
#colon cannot be ignored in df_all.loc[:,cols]
df_all.loc[:,cols]

#People with the same surname may come from different families, for example,
#check the group of surname 'Davies' we found Ticket #48871 corresponds
#to three young men; ticket #33112 corresponds to one women of age 48 and
#and a child with the same surname of age 8. However, an issue is found
#that just using LastName is not sufficient to locate people of the same
#family. For example, the record of the woman with Ticket #33112 shows
#she comes with her two children. By slicing with Ticket #33112, we found the
#woman indeed has two children whose surnames are different. So, we should only
#use Ticket_num instead of LastName to identify people traveling together.
df_all.loc[df_all['LastName']=='Davies',:].sort_values(by=['Ticket_num'])

#Is it possible to have two Mrs in a travelling group? N
#Trick#1: conditinal count after groupby
#The answer is no.
df_all.groupby('Ticket_num')['Ticket'].apply(lambda x: (x=='Mrs').sum()).reset_index(name='bool')



##OK.Above

#todo: with an adult in the group

#todo: check p.104 for missing categorical data

numChildDict = df_all.groupby('Ticket_num')['Age'].apply(lambda x: (x <= cutoffChildAge).sum()).reset_index(name='NumChild')
df_all.join(numChildDict, on = 'Ticket_num')

# the cutoff needs to be tuning in cross-validation
df_all['isChild'] = df_all['Age'] <= 12
df_all.sample(10)



#Who is the parent

# -------------------------------------------------------------------------
##
import random
gs = df_all.groupby('Ticket_num')
type(gs.indices) # dict

#Method 1. Peek the grouped data by sampling; so only part of the data
sampled_group_key = random.sample(gs.groups.keys(), 100)
group_list = list(map(lambda gpkey: gs.get_group(gpkey), sampled_group_key))
for i, g in enumerate(group_list):
    if len(g) > 3:
        print(g)
        break

#We found: the record of the Christy indicates there are two children but only one shown.
df_all.loc[df_all['LastName']=='Christy',:]

#Method 2. scan through the groups
for i, g in gs.groups.items():
    if len(g) > 4:
        print(gs.get_group(i))
        break

#We found: the survival rate of the travelling group is a useful indicator
#It appears none of them are relatives except Lam Ali and Lam Len.
df_all.loc[df_all['Ticket_num']==1601,:]



##Survival rate computation
# The Davies has two children and two adults (one is maid). The youngest child is alive.
df_all.loc[df_all['Ticket_num']==33112,:]


##Identify travelling groups with children among which who are parents?
#Trick#
def addMaxParchMinSibSp(grp):
    return pd.Series(
        [grp['Parch'].max(), grp['SibSp'].min()],
        ['maxParch','minSibSp']
    )
# JOIN versue MERGE [link](https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html)
df_all = df_all.join(df_all.groupby('Ticket_num').apply(addMaxParchMinSibSp), on='Ticket_num')

##
#WRONG df_all.groupby('Ticket_num').apply(lambda x: 1 if x['Parch']==x['maxParch'] else 0, axis=1)
df_all
df_all['isParent'] = np.where((df_all['Parch']==df_all['maxParch']) & (df_all['SibSp']==df_all['minSibSp']), True, False)


##
df_all.loc[df_all['Ticket_num']==36928,:]
df_all['Ticket_num'].value_counts().sample(10)

# The above works. ---------

##
#todo: who is the parents
#todo: sibliing in the same family use their sibling's age to impute missing value
#ToDo: Mother die or Father die and group probability
#ToDo: non-family member
#we assume the most frequent LastName shown in a travel group is the family name.
df_all['Family'] = df_all.groupby(['Ticket_num'])['LastName'].transform(lambda x: x.value_counts().index[0])
#Do we have any babies in the travel group
df_all['hasMaster'] = df_all.groupby(['Ticket_num'])['Title'].transform(lambda x: any(x=='Master'))

#Not all baby can alive
df_all.loc[df_all.Age.eq(1)]

#identify who's with a baby and who's a baby COLMN OPERATIONS.
#Method 1.
# df_all['isMother'] = False
# df_all['isMother'] = df_all['isMother'].where((df_all['Title'] != 'Mrs') | (df_all['hasMaster'] != True), True)
#Method 2.
df_all['MotherWithMaster'] = np.where((df_all['Title']=='Mrs') & (df_all['hasMaster']==True), True, False)

df_all.loc[df_all['Title']=='Master',:][['Age','Survived']].mean() # 5.48, 57%

#Method 3.
def MWM(df):
    return df.apply(lambda x: 1 if np.logical_and(x['Title']=='Mrs', x['Sex']=='female') else 0, axis=1)
df_all['test'] = MWM(df_all)
df_all.head(10)

#Mark any group travelling with mother and children
#Method 1. use transform
temp1 = df_all.groupby(['Ticket_num'])['Title'].transform(lambda x: x.eq('Master').any())
temp2 = df_all.groupby(['Ticket_num'])['MotherWithMaster'].transform(lambda x: x.eq(True).any())
df_all['GroupWMomChild'] = temp1 & temp2

#Method 2-A. use apply-turned dictionary and map IT WORKS!!
#temp5 = df_all.groupby('Ticket_num').apply(lambda x: x['Title'].eq('Master').any() & x['MotherWithMaster'].eq(True).any())
#df_all['GroupWMomChild_3'] = df_all['Ticket_num'].map(temp5)

#Method 2-B. use apply and merge IT WORKS!!
#temp3 = df_all.groupby(['Ticket_num']).apply(lambda x: x['Title'].eq('Master').any())
#temp4 = df_all.groupby(['Ticket_num']).apply(lambda x: x['MotherWithMaster'].eq(True).any())
#df_all.merge((temp3 & temp4).reset_index(), how='left').rename(columns={0: 'GroupWMomChild_2'})

#Male Survival rate is higher for travelling group with mother and children
df_all.groupby(['GroupWMomChild','Pclass','Sex'])['Survived'].mean()



##
#Why the difference? Average age of family without children are older?
df_all.groupby(['GroupWMomChild','Pclass','Sex'])['Age'].mean()

g = sns.FacetGrid(df_all, col='Pclass', row='Sex')
g = g.map(sns.countplot, 'Age')
plt.show()
# g.set(xticks=[range(10,70,10)])
# [plt.setp(ax.get_xticklabels(), rotation=90) for ax in g.axes.flat]
##plt.show()

##
df_all['Age'].dtype

##
#Generally it is the case.
df_all.groupby(['GroupWMomChild','Pclass','Deck','Sex'])['Age'].mean()

#
slice = df_all['GroupWMomChild']==False & df_all['Pclass']==1

##
#Sanity check: is there any group having more than one moether?
df_all.groupby(['Ticket_num'])['MotherWithMaster'].count().sort_values()



## Family group
df_all.loc[df_all['Ticket_num']==3101295,:]
df_all['Ticket_num'].value_counts().head(10)
df_all.groupby(['Ticket_num'])['MotherWithMaster'].count().sort_values(ascending=False)
df_all.loc[df_all['Ticket_num']==2144,:]

##

#Ticekt_num is more useful than LastName (aka Family)
#ToDo: Sensitivity Analysis can have lots of information!!
#ToDo: check to see missing Cabin value (before imputed) leads to higher morality rate? Ticket#347088
#ToDo: does our group have master? does our group have servant?
#ToDo: check the survival rate of master. See ticket#347077, not every master is alive.
#ToDo: even masters with parents die. See ticket#347082
#ToDo: Missing Deck can be "imputed" from travel group (Ticket_num). Ticket#17608
#ToDo: Similarly, missing cabin of servant is not the same as their employer. Ticket#17608
#ToDo: Calculate the family survival rate is a good idea. Inspect it on training data.
#ToDo: Calculate the family survival rate to infer the other femail's survival rate
#ToDo: even in the same family, when women are alive, men are not necessarily alive. Ticket#19950
#ToDo: friends or colleauges are probably in the same ticket number group
#ToDo: group (ticket number) survival rate should have two indicators by sex (Ticket#14879)
#ToDo: do people that travel alone have higher survival rates?

##

#Ticket Number Distribution by Pclass and Embarked
#The plot doesn't help to impute the two missing Embarked value, both of which are in Pclass = 1.
#The only information gain is that given they share the same ticket number, they should know each
#other and highly likely embark from either C or S together.
g = sns.FacetGrid(df_train, col='Pclass', row='Embarked')
g = g.map(sns.countplot, 'Ticket_num')
plt.show()

# #Add percentage bar number 1

#
idx = df_train[df_train['Deck'] == 'T'].index
df_train.loc[idx, 'Deck'] = 'A'

# #Add percentage bar number 2

# #Family size

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
g = sns.FacetGrid(df_train, col='Pclass', row='Embarked', hue='Deck')
g = g.map(plt.scatter, 'Age', 'Fare')
g.add_legend()
plt.show()

#Only two missing Embarked values. Both are female from class 1 and share the same ticket number.
#Sort by age
print(df_train.loc[df_train['Embarked'].isnull(),].sort_values(by=['Age'], ascending=False))

##
# IMPUTATION: EMBARKED
#They more likely board on the ship at port S -- Theory 1.
df_train.loc[df_train['Ticket_num'].between(100000,125000)]['Embarked'].value_counts() # S
df_train.loc[df_train['Fare'].between(60,100)]['Embarked'].value_counts() # S
df_train['Embarked'] = df_train['Embarked'].fillna('S')

# IMPUTATION: AGE
#Impute missing age by sex and class group
df_train['Age'] = df_train.groupby(['Sex','Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

#Distribute of minor; the group with higher young people has higher mortality rate
cols = ['Deck','Pclass']
df_train.groupby(cols).filter(lambda x: x['Age'].quantile(q=0.75) > 50)['Survived'].mean()
df_train.groupby(cols).filter(lambda x: x['Age'].quantile(q=0.75) < 30)['Survived'].mean()


# #Plot training set survival distribution
# https://i.postimg.cc/25rVKwxB/1590377048.png
# https://python-graph-gallery.com/13-percent-stacked-barplot/

# #Categorical variable plot

# #Continuous variable plot

# #Fare binning with qcut or cut

