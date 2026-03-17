from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

sentences = [
    "python programming language",
    "machine learning models",
    "cricket player practise"
]

vec = TfidfVectorizer(max_features=3) 
data = vec.fit_transform(sentences)

k = KMeans(n_clusters=2, n_init=10) 
cluster = k.fit_predict(data)

print("Cluster Results:\n")
for i in range(len(sentences)):
    print(f"Sentence: {sentences[i]}")
    print(f"Cluster: {cluster[i]}")
    print()
