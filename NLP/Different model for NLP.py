# Natural Language Processing

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
ps = PorterStemmer()

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) # Do not remove words which contain a-z and A-Z characters
                                                          # Remove other characters and replace them with Space -> ' '
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)

#print(corpus)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training Different classifiers models on the Training set
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


classifiers = [
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    KNeighborsClassifier(n_neighbors=5),
    DecisionTreeClassifier(random_state=0),
    SVC(gamma='auto'),
    RandomForestClassifier(random_state=0)
    ]

for i in range(len(classifiers)):
    classifiers[i].fit(X_train, y_train)

# Predicting the Test set results
y_pred = []
for i in range(len(classifiers)):
    y_pred.append(classifiers[i].predict(X_test))


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
for i in range(len(classifiers)):
    #cm = confusion_matrix(y_test, y_pred[i])
    #print(cm)
    print('Accuracy for model ' + str(type(classifiers[i])) + ': ' + str(accuracy_score(y_test, y_pred[i])))