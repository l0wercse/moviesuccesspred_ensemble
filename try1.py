#Import all the necessary packages
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


#Read the data of the csv onto a variable and add a new column
data=pd.read_csv('movie_metadata.csv')

bins = [1, 3, 6, 10]
labels = ['FLOP', 'AVG', 'HIT']
data['imdb_binned'] = pd.cut(data['imdb_score'], bins=bins, labels=labels)

data.groupby(['imdb_binned']).size().plot(kind="bar",fontsize=14)
plt.xlabel('Categories')
plt.ylabel('Number of Movies')
plt.title('Categorization of Movies')

data.isnull().sum()

data.dropna(inplace=True)

data.drop(columns=['movie_title','movie_imdb_link'],inplace=True)

le = LabelEncoder()
cat_list=['color', 'director_name', 'actor_2_name',
        'genres', 'actor_1_name',
        'actor_3_name',
        'plot_keywords',
        'language', 'country', 'content_rating',
       'title_year', 'aspect_ratio']
data[cat_list]=data[cat_list].apply(lambda x:le.fit_transform(x))

corr = data.corr()
mask = np.zeros(corr.shape, dtype=bool)
mask[np.triu_indices(len(mask))] = True
plt.subplots(figsize=(20,15))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,cmap='RdYlGn',annot=True,mask = mask)

data.drop(columns=['cast_total_facebook_likes','num_critic_for_reviews'],inplace=True)
data.drop(columns=['imdb_score'],inplace=True)

X = data.iloc[:, 0:23].values
y = data.iloc[:, 23].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0,stratify = y)
print(X_train.shape)
print(y_train.shape)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#KNN WITH BAGGING
knn_model = KNeighborsClassifier(n_neighbors=5)

bagging_model = BaggingClassifier(base_estimator=knn_model, n_estimators=10, random_state=42)
bagging_model.fit(X_train, y_train)
print("----------------KNN BAGGING----------------")


y_pred_knn_bag = bagging_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_knn_bag)
print('\nAccuracy:', accuracy)
print("\n")

cm = confusion_matrix(y_test, y_pred_knn_bag)
print(cm)
print("\n")

cr = classification_report(y_test, y_pred_knn_bag)
print(cr)
print("\n")

#KNN WITH BOOSTING
boosted_knn = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=50, subsample=0.5, max_depth=3, random_state=42)
boosted_knn.fit(X_train, y_train)

print("----------------KNN BOOSTING----------------")

y_pred_knn_boost = boosted_knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_knn_boost)
print('\nAccuracy:', accuracy)
print("\n")

cm = confusion_matrix(y_test, y_pred_knn_boost)
print(cm)
print("\n")

cr = classification_report(y_test, y_pred_knn_boost)
print(cr)
print("\n")

clf_rf=RandomForestClassifier(random_state=0)
rfecv=RFECV(estimator=clf_rf, step=1,cv=5,scoring='neg_log_loss')
rfecv=rfecv.fit(X_train,y_train)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
print('Optimal number of features :', rfecv.n_features_)
print('Best features :', X_train.columns[rfecv.support_])

clf_rf = clf_rf.fit(X_train,y_train)
importances = clf_rf.feature_importances_

std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.title("Log loss vs Number of fetures")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

X_opt = X_train.iloc[:,X_train.columns[rfecv.support_]]
X_test = X_test.iloc[:,X_test.columns[rfecv.support_]]

dset = pd.DataFrame()
data1 = data
data1.drop(columns=['imdb_binned'],inplace=True)
dset['attr'] = data1.columns

dset['importance'] = clf_rf.feature_importances_
dset = dset.sort_values(by='importance', ascending=True)

plt.figure(figsize=(16, 14))
plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Importance', fontsize=14, labelpad=20)
plt.show()

#RANDOM FOREST
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_opt, y_train)

y_pred_randfor = classifier.predict(X_test)

print("----------------RANDOM FOREST----------------")

accuracy = accuracy_score(y_test, y_pred_randfor)
print('\nAccuracy:', accuracy)
print("\n")


cm = confusion_matrix(y_test, y_pred_randfor)
print(cm)
print("\n")

cr = classification_report(y_test, y_pred_randfor)
print(cr)
print("\n")

