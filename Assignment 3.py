# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 07:53:14 2026
The follwing code was created to produce reuslts for clustering and fitting 
99 randomly selected countries were choosen with the varibles relating to 
co2 emissions being urban population, GDP per capita and  Renewable enegry use 
percent. 

@author: Mukuka Mfula
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as skdat
import sklearn.metrics as skmet
import sklearn.preprocessing as pp
import sklearn.cluster as cluster
import scipy.optimize as opt
def one_silhoutte(xy, n):
 """ Calculates silhoutte score for n clusters """
 # set up the clusterer with the number of expected clusters
 kmeans = cluster.KMeans(n_clusters=n, n_init=20)
 # Fit the data, results are stored in the kmeans object
 kmeans.fit(xy) # fit done on x,y pairs
 labels = kmeans.labels_
 # calculate the silhoutte score
 score = (skmet.silhouette_score(xy, labels))
 return score
#%%for the year 2000
#reading the data in a dataframe
df_countries = pd.read_csv("country.csv")
#describing the data 
df_countries.describe()
#making sure the column names are strings
df_countries.columns = df_countries.columns.str.strip()
#producing the correlation figures
corr = df_countries.corr(numeric_only = True)
#displaying the correlation figures to 4 decimal places 
print(corr.round(4))
#creating graphs to see which one makes the best fit for clustering 
plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.show()
pd.plotting.scatter_matrix(df_countries,figsize=(10,10), s=10)
plt.show()
df_countries_copy = df_countries[["CO2 emissions/Per Capita","URBAN POPULATION"]]
scaler = pp.RobustScaler()
#picking the two variables for clustering 
df_clust2 = df_countries[["CO2 emissions/Per Capita","URBAN POPULATION"]]
#fititing it into the scalar 
scaler.fit(df_clust2)
df_norm = scaler.transform(df_clust2)
plt.figure()
plt.scatter(df_norm[:,0],df_norm[:,1],10,marker="o")
plt.show()
#using sihouette score to decide which the number of clusters
for ic in range(2, 11):
 score = one_silhoutte(df_clust2, ic)
 print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")
kmeans = cluster.KMeans(n_clusters=2, n_init=99)
#ftting the data from the results
kmeans.fit(df_norm) 
#getting the cluster labels
labels = kmeans.labels_
#getting the estimated cluster centers
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]
#getting the x and y vaules
x = df_clust2["CO2 emissions/Per Capita"]
y = df_clust2["URBAN POPULATION"]
plt.figure(figsize=(8.0, 8.0))
plt.scatter(x, y, 10, labels, marker="o")
#shwowing the cluster centers
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")
plt.xlabel("CO2 emissions")
plt.ylabel("Urban population")
plt.show()
#%%
#for the year 2020
df_countries2020 = pd.read_csv("countries2020.csv")
#describing the data 
df_countries2020.describe()
#creating correlations for the data 
corr_2 = df_countries2020.corr(numeric_only=True)
#displaying the correlations to four decimal places
print(corr_2.round(4))
#creating graphs to see the best fit fot clustering 
plt.figure()
plt.imshow(corr_2)
plt.colorbar()
plt.show()
pd.plotting.scatter_matrix(df_countries2020,figsize=(10,10), s=10)
plt.show()
#slicing the two columns needed for clustering
df_clust = df_countries2020[["CO2 emissions/Per Capita","URBAN POPULATION"]]
#fitting and transforming the data 
scaler.fit(df_clust)
df_norm2 = scaler.transform(df_clust)
#making the scattered figure
plt.figure()
plt.scatter(df_norm2[:,0],df_norm2[:,1],10,marker="o")
plt.show()
#using sihouette score to decide which the number of clusters 
for ic in range(2, 11):
 score = one_silhoutte(df_clust, ic)
 print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")

kmeans = cluster.KMeans(n_clusters=2, n_init=20)
#ftting the data from the results
kmeans.fit(df_norm2)
#getting the cluster labels
labels = kmeans.labels_
#getting the estimated cluster centers
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]
#getting the x and y vaules 
x = df_clust["CO2 emissions/Per Capita"]
y = df_clust["URBAN POPULATION"]
plt.figure(figsize=(8.0, 8.0))
plt.scatter(x, y, 10, labels, marker="o")
#shwowing the cluster centers
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")
plt.xlabel("CO2 emissions")
plt.ylabel("Urban population")
plt.show()
#%%

df_UK_commisions = pd.read_csv("UK_co2emissions.csv")
df_UK_commisions.plot("YEAR","CO2 Emissions")
plt.show()
def exponential(t, n0, g):
 """Calculates exponential function with scale factor n0 and growth rate g."""
#makes it easier to get a guess for initial parameters
 t = t - 1990
 f = n0 * np.exp(g*t)
 return f
param, covar = opt.curve_fit(exponential, df_UK_commisions["YEAR"], 
                             df_UK_commisions["CO2 Emissions"])
print("CO2 Emissions", param[0]/1e9)
print("decline rate", param[1])
plt.figure()
plt.plot(df_UK_commisions["YEAR"], exponential(df_UK_commisions["CO2 Emissions"], 1.2e12, 0.03))
plt.plot(df_UK_commisions["YEAR"], df_UK_commisions["CO2 Emissions"])
plt.xlabel("Year")
plt.legend()
plt.show()