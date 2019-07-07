from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
text = ["London Paris London","Paris Paris London"]

cv = CountVectorizer()

count_matrix = cv.fit_transform(text)
print (count_matrix.toarray())




Similarity = cosine_similarity(count_matrix.toarray())
print (Similarity)