"""
Created on Tue Apr 18 2017
@author: rezakhoie

"""

import pandas as pd
from scipy.spatial.distance import cosine
import numpy as np
import math

#...Read data CSV file...
data = pd.read_csv('C:/Users/reza9/Desktop/Data Science Repo/Data-Science-Repo/Collaborative Filtering/user-item-rating-sparse.csv')

#...Print top 10 users' ratings...
data.head(10)

#...Keep users' name
names=data.ix[:,0]

#...Drop user names column...
data_itcf=data.drop('user',axis=1)

#.......Some users generally tend to rate high, while some other are opposite.......
#.So we need to normalize ratings, by subtracting average of the user from rating,..
#.........then devide the result by standard deviation of user's ratings............

#...Calculate the average rating of each user...
means=np.mean(data_itcf,axis=1)
#...Calculate the standard deviation of each user's ratings...
sds=np.std(data_itcf,axis=1)

#...Replace ratings with normalized ratings...
for i in range(0,data_itcf.shape[0]) :
    for j in range(0,data_itcf.shape[1]) :
      if not math.isnan(data_itcf.ix[i,j]):
          data_itcf.ix[i,j] = (data_itcf.ix[i,j]-means[i])/sds[i]

#...Replace NaN values with zeros...
data_itcf=data_itcf.fillna(0)

#...Create an empty data frame for users similarities...
data_itcf_similarity = pd.DataFrame(index=names,columns=names)

#...Calculating the cosine similarity of each user with other users...
#........This is useful for user-based collaborative filtering........
for i in range(0,data_itcf.shape[0]) :
    for j in range(0,data_itcf.shape[0]) :
      data_itcf_similarity.ix[i,j] = 1-cosine(data_itcf.ix[i,:],data_itcf.ix[j,:])

#...Replace zeros (unrated items) with weighted average of other users' ratings...      
for m in range(0,data_itcf.shape[0]) :
    for n in range(0,data_itcf.shape[1]) :
        if data_itcf.ix[m,n]==0:
            sum1=0
            similarity_sum=0
            for i in range(0,data_itcf.shape[0]) :
                if (i==m) or (data_itcf.ix[i,n]==0):
                    continue
                else:
                    sum1+=data_itcf_similarity.ix[i,m]*data_itcf.ix[i,n]
                    similarity_sum+=data_itcf_similarity.ix[i,m]
            data_itcf.ix[m,n]=sum1/similarity_sum

#...denormalize ratings by inverse operation to get real-world ratings...
for i in range(0,data_itcf.shape[0]) :
    for j in range(0,data_itcf.shape[1]) :
      if not math.isnan(data_itcf.ix[i,j]):
          data_itcf.ix[i,j] =  data_itcf.ix[i,j]*sds[i] + means[i]

#print the result
print(data_itcf)
            