import pandas as pd
import numpy as np
from sklearn. feature_extraction. text import CountVectorizer
from sklearn. model_selection import train_test_split
from sklearn. tree import DecisionTreeClassifier
import nltk
import re
import pickle
nltk. download('stopwords')
from nltk. corpus import stopwords
stopword=set(stopwords.words('english'))
stemmer = nltk. SnowballStemmer("english")
data = pd. read_csv("Data.csv")
data["labels"] = data["class"]. map({0: "Hate Speech", 1: "Offensive Speech", 2: "No Hate and Offensive Speech"})
data = data[["tweet", "labels"]]
def clean (text):
    text = str (text). lower()
    text = re. sub('[.?]', '', text) 
    text = re. sub('https?://\S+|www.\S+', '', text)
    text = re. sub('<.?>+', '', text)
    text = re.sub(r'[^\w\s]','',text)
    text = re. sub('\n', '', text)
    text = re. sub('\w\d\w', '', text)
    text = [word for word in text.split(' ') if word not in stopword] 
    text=" ". join(text)
    text = [stemmer. stem(word) for word in text. split(' ')]
    text=" ". join(text)
    return text
data["tweet"] = data["tweet"]. apply(clean)
x = np. array(data["tweet"])
y = np. array(data["labels"])
cv = CountVectorizer()
X = cv. fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred = dt.predict (X_test)
y_pred
from sklearn. metrics import accuracy_score
print (accuracy_score (y_test,y_pred))
pickle.dump(dt,open('Hate_Pred','wb'))

pickle.dump(cv,open('c_vectorization','wb'))
