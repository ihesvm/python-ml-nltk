import nltk
import random
from sklearn.svm import SVC
from fastapi import FastAPI
from nltk.corpus import names
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

app = FastAPI()
#
#
# @app.get("/")
# async def root():
#     return {"message": "Hello World"}
#
#
# @app.get("/hello/{name}")
# async def say_hello(name: str):
#     return {"message": f"Hello {name}"}

# Importing the necessary libraries

# Loading the names dataset from NLTK
nltk.download('names')
male_names = [(name, 'male') for name in names.words('male.txt')]
female_names = [(name, 'female') for name in names.words('female.txt')]
labeled_names = male_names + female_names

# Shuffling the labeled names
random.shuffle(labeled_names)


# Extracting features from the names
def gender_features(name):
    features = {}
    features['first_letter'] = name[0].lower()
    features['last_letter'] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features['count({})'.format(letter)] = name.lower().count(letter)
    return features


# Creating the feature sets
featuresets = [(gender_features(name), gender) for (name, gender) in labeled_names]

# Splitting the data into training and testing sets
train_set, test_set = train_test_split(featuresets, test_size=0.2, random_state=42)

# Vectorizing the feature sets
vectorizer = DictVectorizer()
train_features = vectorizer.fit_transform([features for (features, gender) in train_set])
train_labels = [gender for (features, gender) in train_set]

# Training the Support Vector Machine (SVM) model
svm = SVC()
svm.fit(train_features, train_labels)


# Defining the API endpoint
@app.get("/predict/{name}")
def predict_gender(name: str):
    features = gender_features(name)
    features_vector = vectorizer.transform([features])
    prediction = svm.predict(features_vector)[0]
    return {"name": name, "gender": prediction}
