import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

# =============================================================================
# IMPORT DATABASE
# =============================================================================


my_path = r'C:\Users\TITA\OneDrive\Faculdade\2 Mestrado\1º semestre\Data Mining\Project\DataMiningMaster\insurance.db'
#my_path = r'C:\Users\Sofia\OneDrive - NOVAIMS\Nova IMS\Mestrado\Cadeiras\Data mining\Project\DataMiningMaster\insurance.db'
#my_path = r'C:\Users\anacs\Documents\NOVA IMS\Mestrado\Data Mining\Projeto\insurance.db'


# Connect to the database
conn = sqlite3.connect(my_path)

del my_path

cursor = conn.cursor()

# See which tables exist in the database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())


# Getting LOB table
LOB="""
select *
from LOB;
"""
df_lob = pd.read_sql_query(LOB,conn)

del LOB

# Getting Engage table
Engage="""
select *
from Engage;
"""
df_engage = pd.read_sql_query(Engage,conn)

print(list(df_lob.columns.values))
print(list(df_engage.columns.values))

del Engage

# Joining both tables and renaming columns
insurance= """
select L.[Customer Identity] as Cust_ID, [First Policy´s Year] as First_Year, [Brithday Year] as Birthday, [Educational Degree] as Education,
[Gross Monthly Salary] as Monthly_Salary, [Geographic Living Area] as Area, [Has Children (Y=1)] as Children, [Customer Monetary Value] as CMV, [Claims Rate] as Claims_Rate,
[Premiums in LOB: Motor] as Motor, [Premiums in LOB: Household] as Household, [Premiums in LOB: Health] as Health, [Premiums in LOB:  Life] as Life, 
[Premiums in LOB: Work Compensations] as Work_Compensation
from LOB L, Engage E 
where L.[Customer Identity]=E.[Customer Identity]
"""

df_insurance = pd.read_sql_query(insurance,conn)

del df_engage, df_lob, insurance

df_insurance.dtypes

# =============================================================================
# CLEANING DATA
# =============================================================================

# See duplicates  - we found 3 duplicates
duplicated = df_insurance[df_insurance.duplicated(subset = df_insurance.columns[1:])]
df_insurance.shape[0]

# Delete duplicates
df_insurance.drop_duplicates(subset = df_insurance.columns[1:], inplace = True)
df_insurance.shape[0]

del duplicated

#Descriptive statistics
descriptive = df_insurance.describe().T
descriptive['Nulls'] = df_insurance.shape[0] - descriptive['count']

#change education to numeric
df_insurance.Education=df_insurance.Education.str.extract('(\d+)')
df_insurance['Education'] = pd.to_numeric(df_insurance['Education'])

# =============================================================================
# COHERENCE CHECKS
# =============================================================================

#Coherence check of Birthday
            #df_coherence.Birthday=df_coherence.Birthday.apply(lambda x: None if (x>2016 or x<1896) else x)
df_insurance['Coherence_Birthday']=df_insurance.Birthday.apply(lambda x: 1 if (x>2016 or x<1896) else 0)

#Coherence check of First Year
#First_year between 1896 and 2016
            #df_coherence.First_Year=df_coherence.First_Year.apply(lambda x: None if (x>2016 or x<1896) else x)
df_insurance['Coherence_First1']=df_insurance.First_Year.apply(lambda x: 1 if (x>2016 or x<1896) else 0)
df_insurance = df_insurance[df_insurance.Coherence_First1 != 1]

#First_year bigger or equal than Birthday
            #df_coherence['First_Year']=df_coherence.apply(lambda x: None if x.First_Year<x.Birthday else x.First_Year, axis=1)
df_insurance['Coherence_First2']=df_insurance.apply(lambda x: 1 if x.First_Year<x.Birthday else 0, axis=1)
sum(df_insurance['Coherence_First2']) #1997 observations WTF

#Coherence check for Salary (legal age for working in Portugal is 16)
df_insurance['Coherence_Salary']=df_insurance.apply(lambda x:1 if (2016-x.Birthday<16 and x.Monthly_Salary>0) else 0, axis=1)

#Coherence check for Education (it doesnt make sense to finnish bachelor at 16 or less)
df_insurance['Coherence_Education']=df_insurance.apply(lambda x:1 if (2016-x.Birthday<=16 and x.Education=='3 - BSc/MSc') else 0, axis=1)

#Coherence check for Premiums (can't spend more money than they earn)
def summ(num1, *args):
    total=num1
    for num in args:
        total=total+num
    return total

df_insurance['Coherence_Premiums']=df_insurance.apply(lambda x:1 if (summ(x.Motor, x.Household, x.Life, x.Health, x.Work_Compensation)>(x.Monthly_Salary*12)) else 0, axis=1)
df_insurance = df_insurance[df_insurance.Coherence_Premiums != 1]

del df_insurance['Birthday']

del df_insurance['Coherence_Birthday'], df_insurance['Coherence_First1'], df_insurance['Coherence_First2'], df_insurance['Coherence_Salary'], df_insurance['Coherence_Education'], df_insurance['Coherence_Premiums']


# =============================================================================
# OUTLIERS
# =============================================================================

#Outliers para Salary
fig = px.histogram(df_insurance, x=df_insurance.Monthly_Salary, color_discrete_sequence=['darkseagreen'], template='plotly_white')
fig.show()

fig = px.box(df_insurance, y=df_insurance.Monthly_Salary, color_discrete_sequence=['dimgrey'], template='plotly_white')
fig.show()

outliers_bons = df_insurance[(df_insurance.Monthly_Salary>30000)]
df_insurance=df_insurance[(df_insurance.Monthly_Salary<=30000) | (df_insurance.Monthly_Salary.isnull())] #2 rows dropped
#diff=df_insurance[~df_insurance.index.isin(df_insurance1.index)]


#Outliers para CMV
fig = px.histogram(df_insurance, x=df_insurance.CMV, color_discrete_sequence=['darkseagreen'], template='plotly_white')
fig.show()

fig = px.box(df_insurance, y=df_insurance.CMV, color_discrete_sequence=['dimgrey'], template='plotly_white')
fig.show()
    #Remover o outlier dramático
    
# remover outliers e coloca-los num dataset
outliers_fracos = df_insurance[(df_insurance.CMV<-420)] # 15 rows

df_insurance=df_insurance[(df_insurance.CMV>=-125000) | (df_insurance.CMV.isnull())] #1 row dropped
df_insurance=df_insurance[(df_insurance.CMV>=-20000) | (df_insurance.CMV.isnull())] #5 rows dropped
df_insurance=df_insurance[(df_insurance.CMV>=-11000) | (df_insurance.CMV.isnull())] #1 row dropped
df_insurance=df_insurance[(df_insurance.CMV>=-5000) | (df_insurance.CMV.isnull())] #5 rows dropped
df_insurance=df_insurance[(df_insurance.CMV>=-1000) | (df_insurance.CMV.isnull())] #2 rows dropped
df_insurance=df_insurance[(df_insurance.CMV>=-420) | (df_insurance.CMV.isnull())] #1 row dropped

outliers_bons = outliers_bons.append(df_insurance[(df_insurance.CMV>1500)])
df_insurance = df_insurance[(df_insurance.CMV<=1500) | (df_insurance.CMV.isnull())] #12 rows dropped

outliers_bons = outliers_bons.append(df_insurance[(df_insurance.CMV>1320)])
df_insurance=df_insurance[(df_insurance.CMV<=1320) | (df_insurance.CMV.isnull())] #11 rows dropped

# 38 rows dropped total in this variable


#Outliers para Claims
fig = px.histogram(df_insurance, x=df_insurance.Claims_Rate, color_discrete_sequence=['darkseagreen'], template='plotly_white')
fig.show()

fig = px.box(df_insurance, y=df_insurance.Claims_Rate, color_discrete_sequence=['dimgrey'], template='plotly_white')
fig.show()

# remover outliers e coloca-los num dataset
outliers_fracos = outliers_fracos.append(df_insurance[(df_insurance.Claims_Rate>3)])
df_insurance=df_insurance[(df_insurance.Claims_Rate<=3) | (df_insurance.Claims_Rate.isnull())] #1 row dropped


#Outliers para Motor
fig = px.histogram(df_insurance, x=df_insurance.Motor, color_discrete_sequence=['darkseagreen'], template='plotly_white')
fig.show()

fig = px.box(df_insurance, y=df_insurance.Motor, color_discrete_sequence=['dimgrey'], template='plotly_white')
fig.show()

outliers_bons = outliers_bons.append(df_insurance[(df_insurance.Motor>2000)])
df_insurance=df_insurance[(df_insurance.Motor<=2000) | (df_insurance.Motor.isnull())] #3 rows dropped


#Outliers para Health
fig = px.histogram(df_insurance, x=df_insurance.Health, color_discrete_sequence=['darkseagreen'], template='plotly_white')
fig.show()

fig = px.box(df_insurance, y=df_insurance.Health, color_discrete_sequence=['dimgrey'], template='plotly_white')
fig.show()

outliers_bons = outliers_bons.append(df_insurance[(df_insurance.Health>410)])
df_insurance=df_insurance[(df_insurance.Health<=410) | (df_insurance.Health.isnull())] # 5 rows dropped


#Outliers para Life
fig = px.histogram(df_insurance, x=df_insurance.Life, color_discrete_sequence=['darkseagreen'], template='plotly_white')
fig.show()

fig = px.box(df_insurance, y=df_insurance.Life, color_discrete_sequence=['dimgrey'], template='plotly_white')
fig.show()


outliers_bons = outliers_bons.append(df_insurance[(df_insurance.Life>331)])
df_insurance=df_insurance[(df_insurance.Life<=331) | (df_insurance.Life.isnull())] # 6 rows dropped


outliers_bons = outliers_bons.append(df_insurance[(df_insurance.Life>285)])
df_insurance=df_insurance[(df_insurance.Life<=285) | (df_insurance.Life.isnull())] # 11 rows dropped

#Outliers para Household
fig = px.histogram(df_insurance, x=df_insurance.Household, color_discrete_sequence=['darkseagreen'], template='plotly_white')
fig.show()

fig = px.box(df_insurance, y=df_insurance.Household, color_discrete_sequence=['dimgrey'], template='plotly_white')
fig.show()

outliers_bons = outliers_bons.append(df_insurance[(df_insurance.Household>1700)])
df_insurance=df_insurance[(df_insurance.Household<=1700) | (df_insurance.Household.isnull())] # 5 rows dropped


outliers_bons = outliers_bons.append(df_insurance[(df_insurance.Household>1240)])
df_insurance=df_insurance[(df_insurance.Household<=1240) | (df_insurance.Household.isnull())] # 20 rows dropped

#Outliers para Work Compensations
fig = px.histogram(df_insurance, x=df_insurance.Work_Compensation, color_discrete_sequence=['darkseagreen'], template='plotly_white')
fig.show()

fig = px.box(df_insurance, y=df_insurance.Work_Compensation, color_discrete_sequence=['dimgrey'], template='plotly_white')
fig.show()

# remover e append
outliers_bons = outliers_bons.append(df_insurance[(df_insurance.Work_Compensation>400)])
df_insurance=df_insurance[(df_insurance.Work_Compensation<=400) | (df_insurance.Work_Compensation.isnull())] # 3 rows dropped


outliers_bons = outliers_bons.append(df_insurance[(df_insurance.Work_Compensation>300)])
df_insurance=df_insurance[(df_insurance.Work_Compensation<=300) | (df_insurance.Work_Compensation.isnull())] # 12 rows dropped


# TOTAL REMOVED 106 (1%)

#Descriptive statistics after outliers removal
descriptive_o = df_insurance.describe().T
descriptive_o['Nulls'] = df_insurance.shape[0] - descriptive_o['count']


grid = sns.PairGrid(data= df_insurance, vars = ['Monthly_Salary', 'CMV', 'Claims_Rate', 'Motor', 'Household',
                                                'Health', 'Life', 'Work_Compensation'], height = 4)
grid = grid.map_upper(plt.scatter, color = 'darkseagreen')
grid = grid.map_diag(plt.hist, bins = 10, color = 'cadetblue')
grid = grid.map_lower(plt.scatter, color = 'darkseagreen')


# =============================================================================
# CORRELATIONS
# =============================================================================
corr = df_insurance.drop(columns=['Cust_ID', 'Area'])
correlacoes = corr.corr(method='spearman')

import matplotlib.pyplot as plt

correlacoes[np.abs(correlacoes)<0.05] = 0

correlacoes = round(correlacoes,1)

f, ax = plt.subplots(figsize=(9, 9))
mask = np.zeros_like(correlacoes, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

mask_annot = np.absolute(correlacoes.values)>=0.6
annot1 = np.where(mask_annot, correlacoes.values, np.full((11,11),""))
cmap = sns.diverging_palette(49, 163, as_cmap=True)
sns.heatmap(correlacoes, mask=mask, cmap=cmap, center=0, square=True, ax=ax, linewidths=.5, annot=annot1, fmt="s", vmin=-1, vmax=1, cbar_kws=dict(ticks=[-1,0,1]))
sns.set(font_scale=1.2)
sns.set_style('white')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


# =============================================================================
# MISSING VALUES TREATMENT
# =============================================================================

from patsy import dmatrices
import statsmodels.api as sm


#Regressão de First_Year com CMV, Education, Area
y,x = dmatrices('First_Year ~ CMV + Education + Area', data = df_insurance, NA_action='drop', return_type='dataframe')
mod = sm.OLS(y,x)
res = mod.fit()
print(res.summary())
# dá 0% de variância explicada -> não há solução para o first_year, vamos eliminar as observações


# =============================================================
# Delete observations where living area and first year area null
# =============================================================
df_insurance.dropna(subset = ['Area'], inplace = True) # 1 row dropped
df_insurance.dropna(subset = ['First_Year'], inplace = True) # 29 rows dropped

# ===================================================
# Replace null values in the Premium variables with 0
# ===================================================
df_insurance['Motor'].fillna(0, inplace = True)
df_insurance['Health'].fillna(0, inplace = True)
df_insurance['Life'].fillna(0, inplace = True)
df_insurance['Work_Compensation'].fillna(0, inplace = True)

#Descriptive statistics after some nulls treatment
descriptive_n = df_insurance.describe().T
descriptive_n['Nulls'] = df_insurance.shape[0] - descriptive_n['count']

correlacoes_n = df_insurance.corr()

# =============================================
# Replace 2 nulls values in Education using KNN
# =============================================

from sklearn.neighbors import KNeighborsClassifier

incomplete = df_insurance.loc[df_insurance.Education.isnull()]

complete = df_insurance.loc[~df_insurance.Education.isnull()]

clf = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean')

trained_model = clf.fit(complete.loc[:,['Motor','Health','Household','Life','Work_Compensation']], complete.loc[:,'Education'])

imputed_values = trained_model.predict(incomplete.drop(columns=['Cust_ID','Education','First_Year','Monthly_Salary','Area','Children','CMV','Claims_Rate']))

temp_df = pd.DataFrame(imputed_values.reshape(-1,1), columns=['Education'])


incomplete = incomplete.drop(columns='Education')

incomplete.reset_index(drop=True, inplace=True)

incomplete = pd.concat([incomplete, temp_df], axis=1, ignore_index=True)

incomplete.columns=['Cust_ID','First_Year', 'Monthly_Salary', 'Area', 'Children', 'CMV', 'Claims_Rate', 'Motor', 'Household','Health', 'Life', 'Work_Compensation', 'Education']

#drop the nulls values in education
df_insurance.dropna(subset = ['Education'], inplace = True) # 2 rows dropped

#concate the observations that had nulls and were imputed
df_insurance = df_insurance.append(incomplete, sort=True) # 2 rows added

# =============================================================
# Replace 34 nulls values in Monthly_Salary using KNN Regressor
# =============================================================

y,x = dmatrices('Monthly_Salary ~ Education + Motor + Household + Children + Life + Work_Compensation', data = df_insurance, NA_action='drop', return_type='dataframe')
mod = sm.OLS(y,x)
res = mod.fit()
print(res.summary())
# dá 36% de variância explicada

from sklearn.neighbors import KNeighborsRegressor


#criar tabela com as observacoes onde salary é null
data_to_reg_incomplete=df_insurance[df_insurance['Monthly_Salary'].isna()]

#criar outra tabela com as observaçoes inversas
data_to_reg_complete=df_insurance[~df_insurance.index.isin(data_to_reg_incomplete.index)]

#criar o regressor
regressor=KNeighborsRegressor(10, weights='distance', metric='euclidean')

#retirar observaçoes onde children é null
data_to_reg_complete=data_to_reg_complete.loc[data_to_reg_complete.Children.notnull()]

#treinar os dados
neigh=regressor.fit(data_to_reg_complete.loc[:,['Education', 'Motor', 'Household', 'Children', 'Life', 'Work_Compensation']], data_to_reg_complete.loc[:,['Monthly_Salary']])

#criar array com os valores previstos
imputed_Salary=neigh.predict(data_to_reg_incomplete.drop(columns=['Monthly_Salary', 'Area', 'CMV', 'Claims_Rate', 'Cust_ID', 'First_Year', 'Health']))

#converter array em coluna e juntar ao df com coluna vazia
temp_df=pd.DataFrame(imputed_Salary.reshape(-1,1), columns=['Monthly_Salary'])
data_to_reg_incomplete=data_to_reg_incomplete.drop(columns=['Monthly_Salary'])
data_to_reg_incomplete.reset_index(drop=True, inplace=True)
data_to_reg_incomplete=pd.concat([data_to_reg_incomplete, temp_df], axis=1, ignore_index=True)

data_to_reg_incomplete.columns=['Area', 'CMV', 'Children', 'Claims_Rate', 'Cust_ID', 'Education', 'First_Year', 'Health', 'Household', 'Life', 'Motor', 'Work_Compensation', 'Monthly_Salary']

#remover do dataframe original todas as observaçoes onde salary é null
df_insurance=df_insurance.loc[df_insurance['Monthly_Salary'].notnull()]
data_to_reg_incomplete = data_to_reg_incomplete[['Area', 'CMV', 'Children', 'Claims_Rate', 'Cust_ID', 'Education', 'First_Year', 'Health', 'Household', 'Life', 'Monthly_Salary', 'Motor', 'Work_Compensation']]

df_insurance= pd.concat([df_insurance, data_to_reg_incomplete])

# =============================================
# Replace 13 nulls values in Children using KNN
# =============================================
from sklearn.neighbors import KNeighborsClassifier

incomplete = df_insurance.loc[df_insurance.Children.isnull()]

complete = df_insurance.loc[~df_insurance.Children.isnull()]

clf = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean')

trained_model = clf.fit(complete.loc[:,['Monthly_Salary','Motor','Household','Health','Life','Work_Compensation']], complete.loc[:,'Children'])

imputed_values = trained_model.predict(incomplete.drop(columns=['Cust_ID','First_Year','Education','Area','Children','CMV','Claims_Rate']))

temp_df = pd.DataFrame(imputed_values.reshape(-1,1), columns=['Children'])


incomplete = incomplete.drop(columns='Children')

incomplete.reset_index(drop=True, inplace=True)

incomplete = pd.concat([incomplete, temp_df], axis=1, ignore_index=True)

incomplete.columns=['Area','CMV','Claims_Rate','Cust_ID','Education','First_Year','Health','Household','Life','Monthly_Salary', 'Motor','Work_Compensation', 'Children']

#drop the nulls values in children
df_insurance.dropna(subset = ['Children'], inplace = True) # 13 rows dropped

#concate the observations that had nulls and were imputed
df_insurance = df_insurance.append(incomplete, sort=True) # 13 rows added


# =============================================================================
# DESCRIPTIVE STATISTICS AFTER DATA TREATMENT
# =============================================================================

descriptive_an = df_insurance.describe().T
descriptive_an['Nulls'] = df_insurance.shape[0] - descriptive_an['count']


del complete, correlacoes, data_to_reg_complete, data_to_reg_incomplete, descriptive
del descriptive_n, descriptive_o, imputed_Salary, imputed_values, incomplete, mask, temp_df, x, y
del mask_annot, top, bottom, annot1


# =============================================================================
# TRANSFORM VARIABLES - NEW ONES
# =============================================================================

df_insurance['Client_Years']=2016-df_insurance['First_Year']

df_insurance['Yearly_Salary']=12*df_insurance['Monthly_Salary']

df_insurance['Total_Premiums']=df_insurance.loc[:,['Motor','Household','Health','Life','Work_Compensation']][df_insurance>0].sum(1)

# DELETE ROWS WHERE TOTAL_PREMIUMS EQUALS 0
df_insurance = df_insurance[df_insurance['Total_Premiums'] != 0] # 12 rows dropped


df_insurance['Effort_Rate']=df_insurance['Total_Premiums']/df_insurance['Yearly_Salary']

df_insurance['Motor_Ratio']=df_insurance['Motor']/df_insurance['Total_Premiums']
df_insurance['Motor_Ratio']=df_insurance['Motor_Ratio'].where(df_insurance['Motor_Ratio']>0, 0)

df_insurance['Household_Ratio']=df_insurance['Household']/df_insurance['Total_Premiums']
df_insurance['Household_Ratio']=df_insurance['Household_Ratio'].where(df_insurance['Household_Ratio']>0, 0)

df_insurance['Health_Ratio']=df_insurance['Health']/df_insurance['Total_Premiums']
df_insurance['Health_Ratio']=df_insurance['Health_Ratio'].where(df_insurance['Health_Ratio']>0, 0)

df_insurance['Life_Ratio']=df_insurance['Life']/df_insurance['Total_Premiums']
df_insurance['Life_Ratio']=df_insurance['Life_Ratio'].where(df_insurance['Life_Ratio']>0, 0)

df_insurance['Work_Ratio']=df_insurance['Work_Compensation']/df_insurance['Total_Premiums']
df_insurance['Work_Ratio']=df_insurance['Work_Ratio'].where(df_insurance['Work_Ratio']>0, 0)


df_insurance['Negative']=df_insurance.iloc[:,7:13][df_insurance<0].sum(1)

#df_insurance['PayedAdvance_Ratio']=abs(df_insurance['negative'])/df_insurance['Total_Premiums']
df_insurance['Cancelled']=np.where(df_insurance['Negative']<0, 1, 0)

df_insurance['Negative']=abs(df_insurance['Negative'])


df_insurance = df_insurance.drop(columns=['Monthly_Salary','First_Year'])

# SQRT's from original variables with long tails
df_insurance['Life_sqrt'] = np.sqrt(df_insurance['Life']+abs(df_insurance['Life'].min()))
df_insurance['Work_sqrt'] = np.sqrt(df_insurance['Work_Compensation']+abs(df_insurance['Work_Compensation'].min()))
df_insurance['Household_sqrt'] = np.sqrt(df_insurance['Household']+abs(df_insurance['Household'].min()))

# SQRT's from new variables with long tails
#df_insurance['Total_Premiums_sqrt'] = np.sqrt(df_insurance['Total_Premiums']) não vale a pena
df_insurance['Effort_Rate_sqrt'] = np.sqrt(df_insurance['Effort_Rate'])
df_insurance['Life_Ratio_sqrt'] = np.sqrt(df_insurance['Life_Ratio'])
df_insurance['Work_Ratio_sqrt'] = np.sqrt(df_insurance['Work_Ratio']) 
df_insurance['Household_Ratio_sqrt'] = np.sqrt(df_insurance['Household_Ratio'])

df_insurance = df_insurance.drop(columns=['Negative'])

# =============================================================================
# CORRELATIONS WITH NEW VARIABLES
# =============================================================================
corr = df_insurance.drop(columns=['Cust_ID', 'Area'])
correlacoes = corr.corr(method='spearman')
import seaborn as sb
import matplotlib.pyplot as plt

correlacoes[np.abs(correlacoes)<0.05] = 0

correlacoes = round(correlacoes,1)

f, ax = plt.subplots(figsize=(16, 16))
mask = np.zeros_like(correlacoes, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

mask_annot = np.absolute(correlacoes.values)>=0.60
annot1 = np.where(mask_annot, correlacoes.values, np.full((26,26),""))
cmap = sb.diverging_palette(49, 163, as_cmap=True)
sb.heatmap(correlacoes, mask=mask, cmap=cmap, center=0, square=True, ax=ax, linewidths=.5, annot=annot1, fmt="s", vmin=-1, vmax=1, cbar_kws=dict(ticks=[-1,0,1]))
sb.set(font_scale=1)
sb.set_style('white')
bottom, top = ax.get_ylim()             
ax.set_ylim(bottom + 0.5, top - 0.5)

del annot1, mask_annot, bottom, top, mask, corr

# =============================================================================
# STANDARDIZATION
# =============================================================================

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df_std = scaler.fit_transform(df_insurance[['Yearly_Salary','CMV','Claims_Rate','Client_Years',
                                            'Effort_Rate','Effort_Rate_sqrt','Total_Premiums']])
df_std = pd.DataFrame(df_std, columns = ['Yearly_Salary_std','CMV_std','Claims_Rate_std',
                                         'Client_Years_std','Effort_Rate_std',
                                         'Effort_Rate_sqrt_std','Total_Premiums_std'])

df_insurance = df_insurance.join(df_std)


# =============================================================================
# OUTLIERS FOR NEW VARIABLES
# =============================================================================
outliers_fracos = outliers_fracos.append(df_insurance[(df_insurance.Total_Premiums<490)]) 
df_insurance = df_insurance[(df_insurance.Total_Premiums>=490) | (df_insurance.Total_Premiums.isnull())] #34 rows dropped

outliers_bons = outliers_bons.append(df_insurance[(df_insurance.Total_Premiums>1520)]) #14 rows dropped
df_insurance = df_insurance[(df_insurance.Total_Premiums<=1520) | (df_insurance.Total_Premiums.isnull())]


outliers_bons = outliers_bons.append(df_insurance[(df_insurance.Effort_Rate>0.225)]) #31 rows dropped
df_insurance = df_insurance[(df_insurance.Effort_Rate<=0.225) | (df_insurance.Effort_Rate.isnull())]


outliers_bons = outliers_bons.append(df_insurance[(df_insurance.Household_Ratio>0.8)]) #1 row dropped
df_insurance = df_insurance[(df_insurance.Household_Ratio<=0.8) | (df_insurance.Household_Ratio.isnull())]


outliers_bons = outliers_bons.append(df_insurance[(df_insurance.Life_Ratio>0.4)]) #8 rows dropped
df_insurance = df_insurance[(df_insurance.Life_Ratio<=0.4) | (df_insurance.Life_Ratio.isnull())]

outliers_bons = outliers_bons.append(df_insurance[(df_insurance.Work_Ratio>0.4)]) #7 rows dropped
df_insurance = df_insurance[(df_insurance.Work_Ratio<=0.4) | (df_insurance.Work_Ratio.isnull())]

# =============================================================================
# EXPLORATORY ANALYSIS - CATEGORICAL AND NUMERICAL VARIABLES
# =============================================================================
# AREA BOXPLOTS

fig = plt.figure(figsize=(2,3))
fig.set_size_inches(20,10)
fig.subplots_adjust(hspace=0.3, wspace=0.3)

gs = fig.add_gridspec(nrows=2,ncols=3)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[1,0])
ax5 = fig.add_subplot(gs[1,1])
ax6 = fig.add_subplot(gs[1,2])

sns.boxplot(x="Area", y="CMV", data=df_insurance, ax=ax1, color='darkseagreen')
sns.boxplot(x="Area", y="Claims_Rate", data=df_insurance, ax=ax2, color='darkseagreen')
sns.boxplot(x="Area", y="Total_Premiums", data=df_insurance, ax=ax3, color='darkseagreen')
sns.boxplot(x="Area", y="Client_Years", data=df_insurance, ax=ax4, color='darkseagreen')
sns.boxplot(x="Area", y="Yearly_Salary", data=df_insurance, ax=ax5, color='darkseagreen')
sns.boxplot(x="Area", y="Effort_Rate", data=df_insurance, ax=ax6, color='darkseagreen')


df_insurance.drop(columns='Area', inplace=True)

# ALL BOXPLOTS 

fig = plt.figure(figsize=(3,3))
fig.set_size_inches(19,15)
fig.subplots_adjust(hspace=0.4, wspace=0.3)

gs = fig.add_gridspec(nrows=6, ncols=3)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[1,0])
ax5 = fig.add_subplot(gs[1,1])
ax6 = fig.add_subplot(gs[1,2])
ax7 = fig.add_subplot(gs[2,0])
ax8 = fig.add_subplot(gs[2,1])
ax9 = fig.add_subplot(gs[2,2])
ax10 = fig.add_subplot(gs[3,0])
ax11 = fig.add_subplot(gs[3,1])

sns.boxplot(x="Education", y="CMV", hue="Children", data=df_insurance, ax=ax1, palette='BuGn')
ax1.legend(loc='upper right', title='Children')
sns.boxplot(x="Education", y="Claims_Rate", hue="Children", data=df_insurance, ax=ax2, palette='BuGn')
ax2.legend(loc='upper right', title='Children')
sns.boxplot(x="Education", y="Total_Premiums", hue="Children", data=df_insurance, ax=ax3, palette='BuGn')
ax3.legend(loc='upper right', title='Children')
sns.boxplot(x="Education", y="Client_Years", hue="Children", data=df_insurance, ax=ax4, palette='BuGn')
ax4.legend(loc='upper right', title='Children')
sns.boxplot(x="Education", y="Yearly_Salary", hue="Children", data=df_insurance, ax=ax5, palette='BuGn')
ax5.legend(loc='upper right', title='Children')
sns.boxplot(x="Education", y="Effort_Rate", hue="Children", data=df_insurance, ax=ax6, palette='BuGn')
ax6.legend(loc='upper right', title='Children')
sns.boxplot(x="Education", y="Motor_Ratio", hue="Children", data=df_insurance, ax=ax7, palette='BuGn')
ax7.legend(loc='upper right', title='Children')
sns.boxplot(x="Education", y="Household_Ratio", hue="Children", data=df_insurance, ax=ax8, palette='BuGn')
ax8.legend(loc='upper right', title='Children')
sns.boxplot(x="Education", y="Health_Ratio", hue="Children", data=df_insurance, ax=ax9, palette='BuGn')
ax9.legend(loc='upper right', title='Children')
sns.boxplot(x="Education", y="Life_Ratio", hue="Children", data=df_insurance, ax=ax10, palette='BuGn')
ax10.legend(loc='upper right', title='Children')
sns.boxplot(x="Education", y="Work_Ratio", hue="Children", data=df_insurance, ax=ax11, palette='BuGn')
ax11.legend(loc='upper right', title='Children')

## =============================================================================
## CLUSTERING ALGORITHMS
## =============================================================================

# =============================================================================
# K - PROTOTYPES
# =============================================================================

#pip install kmodes
from kmodes.kprototypes import KPrototypes as KP
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster import hierarchy

kproto = df_insurance[['Cust_ID','Yearly_Salary', 'Education', 'Children']]
kproto.reset_index(drop=True, inplace=True)

kp = KP(n_clusters=15, init='Huang', verbose=2)
clusters = kp.fit_predict(VE_Cat[['Yearly_Salary','Education','Children']], categorical=[1,2])

centro=kp.cluster_centroids_
kproto['Label']=kp.labels_
kproto.Label = kproto.Label.apply(pd.to_numeric)

centroids=pd.DataFrame()
centroids['Yearly_Salary']=pd.DataFrame(centro[0]).loc[:,0]
centroids['Education']=pd.DataFrame(centro[1]).loc[:,0]
centroids['Children']=pd.DataFrame(centro[1]).loc[:,1]

del centro, kp, kproto, clusters

Z = linkage(centroids, method = "ward")

hierarchy.set_link_color_palette(['darkseagreen','cadetblue', 'seagreen', 'mediumseagreen', 'c','mediumturquoise','turquoise'])

fig = plt.figure(figsize=(10, 20))
ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2])
Z = hierarchy.linkage(centroids, method='ward')

dendrogram(Z,
           truncate_mode="lastp",
           p=40,
           orientation ="top" ,
           leaf_rotation=45.,
           leaf_font_size=10.,
           show_contracted=True,
           show_leaf_counts=True)
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_visible(False)
cur_axes.axes.get_yaxis().set_visible(False)


ax2.set_xticks([])
ax2.set_yticks([])
ax2.axis('off')

Hclustering = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="ward")
labels = pd.DataFrame(HC.labels_)
labels.columns =  ['Socio-Demo']

df_insurance.reset_index(drop=True, inplace=True)
df_insurance = pd.concat([df_insurance, labels], axis = 1)

del Z, fig, ax2, centroids, cur_axes, Hclustering, labels

fig, axs = plt.subplots(3, 3)
axs[0, 0].hist(df_insurance['Yearly_Salary'].loc[df_insurance['Socio-Demo']==0], color='darkseagreen')
axs[0, 0].set_title('Salary for First Cluster')
axs[0, 1].hist(df_insurance['Education'].loc[df_insurance['Socio-Demo']==0], color='cadetblue')
axs[0, 1].set_title('Education for First Cluster')
axs[0, 2].hist(df_insurance['Children'].loc[df_insurance['Socio-Demo']==0], color='tan')
axs[0, 2].set_title('Children for First Cluster')
axs[1, 0].hist(df_insurance['Yearly_Salary'].loc[df_insurance['Socio-Demo']==1], color='darkseagreen')
axs[1, 0].set_title('Salary for Second Cluster')
axs[1, 1].hist(df_insurance['Education'].loc[df_insurance['Socio-Demo']==1], color='cadetblue')
axs[1, 1].set_title('Education for Second Cluster')
axs[1, 2].hist(df_insurance['Children'].loc[df_insurance['Socio-Demo']==1], color='tan')
axs[1, 2].set_title('Children for Second Cluster')
axs[2, 0].hist(df_insurance['Yearly_Salary'].loc[df_insurance['Socio-Demo']==2], color='darkseagreen')
axs[2, 0].set_title('Salary for Third Cluster')
axs[2, 1].hist(df_insurance['Education'].loc[df_insurance['Socio-Demo']==2], color='cadetblue')
axs[2, 1].set_title('Education for Third Cluster')
axs[2, 2].hist(df_insurance['Children'].loc[df_insurance['Socio-Demo']==2], color='tan')
axs[2, 2].set_title('Children for Third Cluster')
plt.show()

# =============================================================================
# SOM + HIERARCHICAL
# =============================================================================
from sompy.sompy import SOMFactory
from sompy.visualization.plot_tools import plot_hex_map
import logging

X = df_insurance[['Children', 'Education', 'Yearly_Salary']].values

names = ['Children', 'Education', 'Yearly_Salary']
sm = SOMFactory().build(data = X,
               mapsize=(10,10),
               normalization = 'var',
               initialization='random', #'random', 'pca'
               component_names=names,
               lattice='hexa', #'rect','hexa'
               training = 'seq') #'seq','batch'

sm.train(n_job=4, verbose='info',
         train_rough_len=30,
         train_finetune_len=100)

final_clusters = pd.DataFrame(sm._data, columns = ['Children', 'Education', 'Yearly_Salary'])

my_labels = pd.DataFrame(sm._bmu[0])
    
final_clusters = pd.concat([final_clusters,my_labels], axis = 1)

final_clusters.columns = ['Children', 'Education', 'Yearly_Salary','Lables']

from sompy.visualization.mapview import View2DPacked
view2D  = View2DPacked(10,10,"", text_size=7)
view2D.show(sm, col_sz=5, what = 'codebook',) #which_dim="all", denormalize=True)
plt.show()

from sompy.visualization.mapview import View2D
view2D  = View2D(10,10,"", text_size=7)
view2D.show(sm, col_sz=5, what = 'codebook',) #which_dim="all", denormalize=True)
plt.show()

from sompy.visualization.bmuhits import BmuHitsView
vhts  = BmuHitsView(12,12,"Hits Map",text_size=7)
vhts.show(sm, anotate=True, onlyzeros=False, labelsize=10, cmap="autumn", logaritmic=False)


# Apply Hierarchical Clustering in the SOM results

# We need scipy to plot the dendrogram 
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster import hierarchy
from pylab import rcParams

# The final result will use the sklearn
import sklearn
from sklearn.cluster import AgglomerativeClustering

plt.figure(figsize=(10,5))
plt.style.use('seaborn-whitegrid')

#Scipy generate dendrograms
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
Z = linkage(sm._data, method = "ward") # method='single','complete','ward'

#https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
hierarchy.set_link_color_palette(['c', 'm', 'y', 'g','b','r','k'])

dendrogram(Z,
           truncate_mode="lastp",
           p=40,
           orientation ="top" ,
           leaf_rotation=45.,
           leaf_font_size=10.,
           show_contracted=True,
           show_leaf_counts=True)














