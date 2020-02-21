import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
################################################ TASK 2 #############################################################################
 #  DATA PREPROCESSING 

df = pd.read_csv("market.csv")
df = df[['State','Education','EmploymentStatus','Marital Status','Total Claim Amount','Gender','Income','Customer Lifetime Value','Number of Open Complaints','Months Since Policy Inception','Monthly Premium Auto','Location Code','Vehicle Size']]
di = {'Arizona': 10, 'Oregon': 20, 'California': 30, 'Nevada': 40,'Washington': 50}
df['State'] = df['State'].map(di)
di1 = {'Bachelor':1,'Master': 2, 'High School or Below': 3, 'Doctor' : 4, 'College': 5}
df['Education'] = df['Education'].map(di1)
di2 = {'M': 10, 'F':20}
df['Gender'] = df['Gender'].map(di2)
di3 = {'Single' : 1, 'Divorced': 2, 'Married': 3}
df['Marital Status'] = df['Marital Status'].map(di3)
di4 = {'Employed' : 10, 'UnEmployed': 20, 'Disabled': 30, 'Medical Leave' : 40}
df['EmploymentStatus'] = df['EmploymentStatus'].map(di4)
di5 = {'Rural': 1, 'Urban' : 2 , 'Suburban': 3}
df['Location Code'] = df['Location Code'].map(di5)
di6 = {'Medsize': 10 ,'Small': 20 ,'Large':30}
df['Vehicle Size'] = df['Vehicle Size'].map(di6)
df = df.dropna()
df = df.reset_index()
df.to_csv("markt.csv")
df = pd.read_csv("markt.csv")
print(df.head(20))
# X = df.values
# n = 10 
kmeans = KMeans(n_clusters=10, random_state=1).fit(df)
labels = kmeans.labels_
print(davies_bouldin_score(df, labels))	


# the Davies Bouldin Score is 0.8 approximately, the lesser the score the higher effiecient clustering. Since all the parameters are not taken because of time being.  
# The score can be improved for more number of features are included. 

################################################################### TASK 3 #############################################################################3
df =df[['Vehicle Size','Total Claim Amount','Marital Status','Income','Customer']]
conditions = [(df['Total Claim Amount'] > 200) & df['Vehicle Size'] == 10]
df = df[['Customer','Income','Marital Status']]
print(df.head(20))
df.to_csv("output2.csv")

