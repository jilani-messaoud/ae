import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.metrics.distance import jaccard_distance
filename ="data.json"
data = json.load(open(filename, "r"))
print(data)
text1 = data["question"][0]["label"]
text2 = data["question"][9]["label"]
print(text1)
print(text2)
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalpha() and word not in stop_words]

tokens1 = preprocess(text1)
tokens2 = preprocess(text2)
print (tokens1)
print (tokens2)
jaccard_sim = 1 - jaccard_distance(set(tokens1), set(tokens2))
print("Jaccard similarity:", jaccard_sim)
for i in range(len(data["question"])):
    text2 = data["question"][i]["label"]
    tokens2 = preprocess(text2)
    data["question"][i]["distance"]=1 - jaccard_distance(set(tokens1), set(tokens2))
sorted_data = sorted(data["question"], key=lambda x: x["distance"], reverse=True)
print(sorted_data)
with open('table.json', 'w') as json_file:
    json.dump(sorted_data, json_file)
