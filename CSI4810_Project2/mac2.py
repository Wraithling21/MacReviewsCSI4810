import numpy as np
import pandas as pd
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score,f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud, STOPWORDS

from langdetect import detect
nltk.download('punkt')
nltk.download("stopwords")

##NOTE: DEVELOP VISUAL MODELS FOR REPORT AND PRESENTATION

def score_hist(data):
    labels, counts = np.unique(data, return_counts = True)
    plt.bar(labels, counts, align = 'center')
    plt.xlabel("Favorable Vs Unfavorable")
    plt.ylabel("Frequency")
    plt.title("Favorability Chart", loc = "center")


##Used in the TfidfVectorizer analyzer to preprocess our dataset and clean it
analyzer_tf = TfidfVectorizer().build_analyzer()
def clean_tf(doc):
    stopword = nltk.corpus.stopwords.words('english')
    porter = PorterStemmer()
    no_stop_words = [word for word in analyzer_tf(doc) if word not in stopword]
    no_alpha = [word for word in no_stop_words if word.isalpha()]
    stemmed = [porter.stem(w) for w in no_alpha]
    return(stemmed)

def detectLanguage(Reviews):
    Reviews_arr = Reviews.to_numpy()
    Reviews_list = list(Reviews_arr)
    language_type = []
    for review in Reviews_list:
        language = detect(review)
        language_type.append(str(language))
    return language_type

def tfidf_vectorize(data, vect):
    X_tfidf = vect.fit_transform(data)
    feature_names = vect.get_feature_names()
    X_tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns = feature_names)
    return(X_tfidf_df)

def generate_cloud(text):
    stopwords = nltk.corpus.stopwords.words('english')
    porter = PorterStemmer()
    words = " ".join(text)
    tokens = word_tokenize(words)
    lower = [word.lower() for word in tokens]
    no_stop_words = [word for word in lower if word not in stopwords]
    no_alpha = [word for word in no_stop_words if word.isalpha()]
    stemmed = [porter.stem(w) for w in no_alpha]
    words = " ".join(stemmed)
    plt.figure(figsize=(16,13))
    wc = WordCloud(background_color = "white",
                    max_words = 100,
                    max_font_size=50,
                    stopwords =  nltk.corpus.stopwords.words('english'))
    wc.generate(words)

    plt.imshow(wc.recolor(colormap='Pastel2', random_state=17), alpha=1.00)

    plt.show()

def search(X,y):
    rf = RandomForestClassifier(class_weight = "balanced")
    parameters = {
        "n_estimators" : [5,10,15,50,100,128],
        "max_depth" : [2,4,10,20],
    }
    cv = GridSearchCV(rf, parameters)
    cv.fit(X,y.values.ravel())
    print(cv.best_params_)
    print(cv.best_estimator_)

## Reads in the Review File and removes the unncessarly created columns
# pd.set_option('display.max_rows', None)
pd.set_option("max_columns", None)
def loadDataFrame(file):
    df = pd.read_excel(file)
    df = df.dropna()
    return df

##Recreates Dataframe where stars Represent Favorable or Unfavorable reviews, 3 is considered neutral
def processDataFrame(df):
    Reviews = df["Review Content"]
    df_filtered = df.loc[df["Rating"] != 3]
    df_filtered = np.where(df['Rating'] > 3, 1, 0)
    #Creates that new Dataframe with favorable or unfavorable reviews
    new_df = pd.DataFrame(Reviews, columns = ["Review Content"])
    new_df = new_df.assign(Favorable = df_filtered)
    return new_df

df = loadDataFrame("macreviews_2.xlsx")
new_df = processDataFrame(df)

##Takes the ratings and passes them into score_hist, which creates a histogram of star reviews
# ratings = df["Rating"]
# ratings = ratings.to_numpy()
# score_hist(ratings)

##Detect languages, and finally removes any non-english words.
Reviews = new_df["Review Content"]
language_type = detectLanguage(Reviews)
new_df = new_df.assign(language_type = language_type)
new_df = new_df.loc[new_df["language_type"] == "en"]

# favorable = new_df["Favorable"]
# favorable = favorable.to_numpy()
# score_hist(favorable) ## Here we remove 3 as it is neutral, 1,2:0; 4,5:1

X = new_df["Review Content"]
y = new_df["Favorable"]

# generate_cloud(X)
tfidf_vect = TfidfVectorizer(lowercase = True, analyzer = clean_tf)
tfidf_vect_fit = tfidf_vect.fit(X)
X = tfidf_vectorize(X, tfidf_vect_fit)
feature_names_train = tfidf_vect.get_feature_names()

# search(X,y)
##Random Forest Classifier for Favorable or Unfavorable Review Classification
rf = RandomForestClassifier(class_weight = "balanced", max_depth = 4, n_estimators = 128)

##Metrics that were used for determine accuracy
scoring = {'accuracy' : make_scorer(accuracy_score),
           'f1_score' : make_scorer(f1_score)}

scores = cross_validate(rf, X, y.values.ravel(), cv = 5, scoring = scoring)
print(scores)
y_pred = cross_val_predict(rf, X, y.values.ravel(), cv = 5)
conf_matrix = confusion_matrix(y.values.ravel(), y_pred)

ax = sns.heatmap(conf_matrix, annot = True, fmt = "d")

plt.show()
