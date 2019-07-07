import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return data[data.index == index]["title"].values[0]

def get_index_from_title(title):
	return data[data.title == title]["index"].values[0]
##################################################

##Step 1: Read CSV File
data = pd.read_csv("movie_dataset.csv")
#print (data.columns)
##Step 2: Select Features
features = ['keywords','cast','genres','director']
for feature in features :
	data[feature] = data[feature].fillna('')
##Step 3: Create a column in DF which combines all selected features
def combine_features(row):
	try :
		return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']
	except:
		print ("Error:", row)
data["combined_features"] = data.apply(combine_features,axis = 1)
#print (data["combined_features"].head())

##Step 4: Create count matrix from this new combined column

cv = CountVectorizer()

count_matrix = cv.fit_transform(data["combined_features"])
#print (count_matrix)
##Step 5: Compute the Cosine Similarity based on the count_matrix
cosine_similarity = cosine_similarity(count_matrix)
#print(cosine_similarity)

movie_user_likes = "The Avengers"

## Step 6: Get index of this movie from its title

index = get_index_from_title(movie_user_likes)
similar_movies = list(enumerate(cosine_similarity[index]))
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)
 ## Step 7: Get a list of similar movies in descending order of similarity score



## Step 8: Print titles of first 50 movies
i = 0
for movie in sorted_similar_movies :
	print (get_title_from_index(movie[0]))
	i+=1
	if i>50:
		break
	
