#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import geopandas as gpd
get_ipython().run_line_magic('matplotlib', 'inline')
import random
import plotly.io as pio
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.offline as py
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.colors import n_colors
from plotly.subplots import make_subplots
init_notebook_mode(connected=True)


# In[2]:


df=pd.read_csv(r"C:\Users\HIMANSHU KUMAR\OneDrive\Desktop\LS_2.0.csv")
df.head()


# Here i am exploring total head/verticals/variables/features from the top.

# In[3]:


df.tail()


# Here i am exploring total head/verticals/variables/features from the bottom. from here we will know about total number of unit of analysis/rows.

# In[4]:


df.shape


# this talks about  the matrix means total number of unit of analysis/rows and total number of features/variables.

# In[5]:


df.describe


# it describes about data frames( matrices) like from top and bottom.
# generally it provides a summary of the central tendency, dispersion, and shape of the distribution of a DataFrame's numeric columns.

# In[6]:


df.info()


# In[7]:


df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace('\n','_') 
df.rename(columns = {'over total electors _in constituency':'total_voters',
                    'over total votes polled _in constituency':'votes_polled',
                    'total electors':'total_electors'},inplace=True)


# In[8]:


print('Major parties that contested elections in 2019 General Elections =',df['party'].nunique())


# In[9]:


def value_cleaner(x):
    try:
        str_temp = (x.split('Rs')[1].split('\n')[0].strip())
        str_temp_2 = ''
        for i in str_temp.split(","):
            str_temp_2 = str_temp_2+i
        return str_temp_2
    except:
        x = 0
        return x
df['assets'] = df['assets'].apply((value_cleaner))
df['liabilities'] = df['liabilities'].apply((value_cleaner))
df.head()


# In[10]:


df['education']=df['education'].str.replace('\n','')
df['party']=df['party'].str.replace('TRS','BRS')


# In[11]:


df1=df[df['party']!='NOTA']
df1.isnull().sum()


# There are no missing values

# In[12]:


df.dtypes


# The function gives a series with the data type of each column.

# In[13]:


df=df.fillna(0)
df['criminal_cases']=df['criminal_cases'].replace({'Not Available':0})
df['criminal_cases'] = pd.to_numeric(df['criminal_cases'],errors='coerce').astype(np.int64)
df['age']=df['age'].apply(lambda x:round(x))

def numer(i):
    df[i]=pd.to_numeric(df[i])

numer('assets')
numer('liabilities')
numer('age')


# In[14]:


fig,ax=plt.subplots(figsize=(15,15))
sns.heatmap(df.corr(),annot=True)


# here we are visualising correlation using heat map.
# if coefficinet of correlation is 1 then variables are strongly correlated.

# In[15]:


df.head(15)


# # CREATING SUNBURST IMAGES OF ALL STATES AND THEIR CONSTITUENCIES
# 

# In[16]:


st_val=df[['state','constituency','total_electors']]
fig=px.sunburst(st_val,path=['state','constituency']
                ,values='total_electors',
               color='total_electors',
               color_continuous_scale='viridis_r')
fig.update_layout(title_text='Sunburst Image of States and Constituencies',
                  template='plotly_dark')
fig.show()


# Hierarchical data is best displayed with a sunburst chart. One ring or circle is used to represent each level of the hierarchy, with the innermost circle serving as the hierarchy's top level.

# In[17]:


df1=df[df['party']!='NOTA']


# # Finding the gender distribution

# In[18]:


req=pd.DataFrame(df1['gender'])
fig=px.pie(req,names='gender')
fig.update_layout(title_text='Gender Distribution in Participation')
fig.show()


# In[19]:


gen=df1.groupby('gender').apply(lambda x:x['name'].count()).reset_index(name='counts')
gen['category']='Overall gender ratio'
winners=df1[df1['winner']==1]
gen_win=winners.groupby('gender').apply(lambda x:x['name'].count()).reset_index(name='counts')
gen_win['category']='Winning gender ratio'
total=pd.concat([gen_win,gen])

fig=px.bar(total,x='gender',y='counts',color='category',barmode='group')
fig.update_layout(title_text='Participation Vs Win counts')
fig.show()


# In[20]:


party_state=df1.groupby('party')['state'].nunique() .reset_index(name='state')
party_const=df1.groupby('party')['constituency'].count() .reset_index(name='constituency')
top_const=party_const.sort_values(by='constituency',ascending=False)[:25]
top_party=pd.merge(top_const,party_state,how='inner',left_on='party',right_on='party')
fig=px.scatter(top_party,x='constituency',
               y='state',color='state',size='constituency',
               hover_data=['party'])
fig.update_layout(title_text='Constituency vs Statewise participation for the most contesting Political Parties')
fig.show()


# # Statewise performance report of each party

# In[21]:


st_party=winners.groupby(['party','state'])['winner'].sum().reset_index(name='wins')
pivot_st_party=pd.pivot(st_party,index='party',columns='state',values='wins')
plt.figure(figsize=(15,35))
sns.heatmap(pivot_st_party,annot=True,fmt="g")
plt.xlabel('state')
plt.ylabel('party')
plt.title('Statewise report card for the Political Parties in India',size=25)


# # Vote share of top 5  political parties

# In[22]:


df1.head(5)


# In[23]:


vote_share_top5=df1.groupby('party')['total_votes'].sum().nlargest(5).index.tolist()
def vote_share(row):
    if row['party'] not in vote_share_top5:
        return 'other'
    else:
        return row['party']
df1['party_new']=df1.apply(vote_share,axis=1)
counts = df1.groupby('party_new')['total_votes'].sum()
labels = counts.index
values = counts.values
pie = go.Pie(labels=labels, values=values, marker=dict(line=dict(color='#000000', width=1)))
layout = go.Layout(title='Political Partywise Vote Share')
fig = go.Figure(data=[pie], layout=layout)
py.iplot(fig)


# # Relative comparison of winners with criminal case Vs winners with no criminal cases-

# In[24]:


winners1=winners[winners['criminal_cases']!=0]
winners0=winners[winners['criminal_cases']==0]

winners_cri=winners1.groupby('party')['name'].count().reset_index(name='candidates')

winners_no_cri=winners0.groupby('party')['name'].count().reset_index(name='candidates')

winners_cri['status']='pending_case'
winners_no_cri['status']='no_pending_case'

final_winners=pd.concat([winners_no_cri,winners_cri])

fig=px.bar(final_winners,x='party',y='candidates',color='status')
fig.update_layout(title_text='Winners with criminal cases vs no criminal cases in parties')
fig.show()


# # Age Vs Candidates relationship distribution-

# In[25]:


age_cnt=df1.groupby(['age','gender'])['name'].count() .reset_index(name='counts')
fig = px.histogram(age_cnt,x='age',y='counts',color='gender'
              ,marginal='violin',
              title='Age Counts Distribution among the candidates')
fig.update_layout(title_text='Age  Distribution among the candidates')
fig.show()


# # Average age of candidates partywise-

# In[26]:


pt_avg_age=df1.groupby('party')['age'].mean().round() .reset_index(name='avg_age')
final_avg_age=pd.merge(top_party['party'],pt_avg_age,
                       how='inner',left_on='party',
                       right_on='party')
final_avg_age=final_avg_age.sort_values(by='avg_age',ascending=False)

plt.figure(figsize=(10,10))
fig=px.bar(final_avg_age,x='party',y='avg_age',
           color='avg_age')
fig.update_layout(title_text='Average age of candidates in each party')
fig.show()


# In[27]:


winner_avg_age=winners1.groupby('party')['age'].mean().round() .reset_index(name='avg_age')
final_avg_age=pd.merge(top_party['party'],winner_avg_age,
                       how='inner',left_on='party',
                       right_on='party')
final_avg_age=final_avg_age.sort_values(by='avg_age',ascending=False)

plt.figure(figsize=(10,10))
fig=px.bar(final_avg_age,x='party',y='avg_age',
           color='avg_age')
fig.update_layout(title_text='Average age of candidates in each party')
fig.show()


# In[28]:


cri_cases=df1.groupby('criminal_cases')['name'].count().reset_index(name='counts')
fig = px.histogram(cri_cases, x='criminal_cases',y='counts',marginal='violin')
fig.update_layout(title_text='Criminal Cases Counts Distribution among the politicians')
fig.show()


# In[29]:


df1[df1['criminal_cases']==240]


# In[30]:


cat_overall=df1.groupby('category')['name'].count().reset_index(name='counts')
cat_overall['status']='Overall Category Counts'
cat_win=winners.groupby('category')['name'].count().reset_index(name='counts')
cat_win['status']='Winner Category Counts'
cat_overl_win=pd.concat([cat_win,cat_overall])
fig=px.bar(cat_overl_win,x='category',y='counts',
           color='status',barmode='group')
fig.update_layout(title_text='election  vs winning among categories')
fig.show()


# In[31]:


df1['education'].unique()


# In[32]:


df1.head()


# In[33]:


ed_cnt=df1.groupby('education')['name'].count().reset_index(name='counts')
fig=go.Figure(data=[go.Pie(labels=ed_cnt['education'],values=ed_cnt['counts'],
                           pull=[0.1, 0.2, 0, 0.1, 0.2, 0,0.1, 0.1, 0.2,0, 0.1, 0.2],
                           title='Educational Qualification of all the Contesting Candidates')])
                           

fig.update_layout(title_text='Overall Education Qualification of all the Contesting Candidates',
                  template='plotly_dark')
fig.show()

ed_win_cnt=winners.groupby('education').apply(lambda x:x['party'].count()) .reset_index(name='counts')
fig2 = go.Figure(data=[go.Pie(labels=ed_win_cnt['education'], values=ed_win_cnt['counts'], 
                                 pull=[0.1, 0.2, 0, 0.1, 0.2, 0,0.1, 0.1, 0.2,0, 0.1, 0.2],
                              title='Qualification of the Winners')])
fig2.update_layout(title_text='Qualification of the Winners')
fig2.show()


# In[34]:


win_as_liab=winners.sort_values(by='assets',ascending=False)
fig=px.scatter(win_as_liab,x='assets',y='liabilities'
               ,color='state',
               size='assets',
               hover_data=(['name','party','constituency','state','winner']),
                 title='Assets vs Liabilities for the Winning Politicians')
fig.update_layout(title_text='Assets vs Liabilities of the winners')
fig.show()


# In[35]:


df1=df1.reset_index()


# In[36]:


df1['id']=df1.index//3


# In[37]:


df1=df1.drop(['index'],axis=1)


# In[38]:


rev=df1.groupby(['id','constituency','party','category']).apply(lambda x:x['total_votes']).reset_index(name='counts')
rev0=rev.groupby(['id','constituency','category'])['category'].count().reset_index(name='counts')


# In[39]:


rev.columns


# In[40]:


rev=rev.drop(['level_4'],axis=1)


# In[41]:


rev0.shape


# In[42]:


y=[]
for i,j in zip(rev0['counts'],rev0['category']):
    if i==3 and j=='ST':
        y.append(f'Reserved for {j}')
    elif i==3 and j=='SC':
        y.append(f'Reserved for {j}')
    else:
        y.append('General')


# In[43]:


rev0=rev0.assign(status=y)


# In[44]:


rev0.head()


# In[45]:


rev0=rev0.drop_duplicates('constituency')


# In[46]:


rev0.shape


# In[47]:


rev0.groupby('category')['status'].count()


# Estimated values-
# General seats : 402
# Reserved for SC : 84
# Reserved for ST : 53
# 
# 
# 
# 
# Real values-
# General seats : 412
# Reserved for SC : 84
# Reserved for ST : 47
# 
# 
# 
#    
