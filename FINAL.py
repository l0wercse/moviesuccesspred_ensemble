#Import all the necessary packages
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error,r2_score
import warnings
warnings.filterwarnings('ignore')


#Read the data of the csv onto a variable and add a new column
data=pd.read_csv('movie_metadata.csv')
print(data.shape)
bins = [1, 3, 6, 10]
labels = ['FLOP', 'AVG', 'HIT']
data['imdb_binned'] = pd.cut(data['imdb_score'], bins=bins, labels=labels)

data.dropna(inplace=True)

data.drop(columns=['movie_title','movie_imdb_link'],inplace=True)

le = LabelEncoder()
cat_list=['color', 'director_name', 'actor_2_name',
        'genres', 'actor_1_name',
        'actor_3_name',
        'plot_keywords',
        'language', 'country', 'content_rating',
       'title_year', 'aspect_ratio','imdb_binned']
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
plt.title("Log loss vs Number of features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

X_opt = X_train.iloc[:,X_train.columns[rfecv.support_]]
X_test = X_test.iloc[:,X_test.columns[rfecv.support_]]
print(X_opt.columns)
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

#KNN

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_opt, y_train)

print("----------------KNN----------------")

y_pred_knn= knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_knn)
print('\nAccuracy:', accuracy)
print("\n")


cm = confusion_matrix(y_test, y_pred_knn)
print(cm)
print("\n")

cr = classification_report(y_test, y_pred_knn)
print(cr)
print("\n")

#KNN WITH BAGGING
knn_model = KNeighborsClassifier(n_neighbors=5)

bagging_model = BaggingClassifier(base_estimator=knn_model, n_estimators=10, random_state=42)
bagging_model.fit(X_opt, y_train)

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
boosted_knn.fit(X_opt, y_train)

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

#KNN WITH BEST K
k_values = range(1, 21)
best_k = 0
best_accuracy = 0
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    accuracy = sum(cross_val_score(knn, X_train, y_train, cv=10, scoring="accuracy")) / 10
    if accuracy > best_accuracy:
        best_k = k
        best_accuracy = accuracy

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_opt, y_train)

print("----------------KNN WITH BEST K----------------")

y_pred_knn_bestk = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_knn_bestk)
print('\nAccuracy:', accuracy)
print("\n")


cm = confusion_matrix(y_test, y_pred_knn_bestk)
print(cm)
print("\n")

cr = classification_report(y_test, y_pred_knn_bestk)
print(cr)
print("\n")

#DECISION TREES
tree_clf = DecisionTreeClassifier(random_state=42)

tree_clf.fit(X_opt, y_train)
y_pred_dt = tree_clf.predict(X_test)

print("----------------DECISION TREES----------------")

accuracy = accuracy_score(y_test, y_pred_dt)
print('\nAccuracy:', accuracy)
print("\n")


cm = confusion_matrix(y_test, y_pred_dt)
print(cm)
print("\n")

cr = classification_report(y_test, y_pred_dt)
print(cr)
print("\n")

#DECISION TREES WITH BAGGING
tree_clf = DecisionTreeClassifier(random_state=42)
bag_clf = BaggingClassifier(base_estimator=tree_clf, n_estimators=100, max_samples=0.5, max_features=0.5, random_state=42)

bag_clf.fit(X_opt, y_train)
y_pred_dtbag = bag_clf.predict(X_test)

print("----------------DECISION TREES WITH BAGGING----------------")

accuracy = accuracy_score(y_test, y_pred_dtbag)
print('\nAccuracy:', accuracy)
print("\n")


cm = confusion_matrix(y_test, y_pred_dtbag)
print(cm)
print("\n")

cr = classification_report(y_test, y_pred_dtbag)
print(cr)
print("\n")

#DECISION TREES WITH BOOSTING
tree_clf = DecisionTreeClassifier(max_depth=1)
boost_clf = AdaBoostClassifier(base_estimator=tree_clf, n_estimators=100, learning_rate=1, random_state=42)

boost_clf.fit(X_opt, y_train)
y_pred_dtboost = boost_clf.predict(X_test)

print("----------------DECISION TREES WITH BOOSTING----------------")

accuracy = accuracy_score(y_test, y_pred_dtboost)
print('\nAccuracy:', accuracy)
print("\n")


cm = confusion_matrix(y_test, y_pred_dtboost)
print(cm)
print("\n")

cr = classification_report(y_test, y_pred_dtboost)
print(cr)
print("\n")

#RANDOM FOREST
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
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

#RANDOM FOREST WITH BAGGING

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
bag = BaggingClassifier(base_estimator=classifier, n_estimators=10, random_state=42)
bag.fit(X_opt, y_train)

y_pred_randfor_bag = bag.predict(X_test)

print("----------------RANDOM FOREST WITH BAGGING----------------")

accuracy = accuracy_score(y_test, y_pred_randfor_bag)
print('\nAccuracy:', accuracy)
print("\n")


cm = confusion_matrix(y_test, y_pred_randfor_bag)
print(cm)
print("\n")

cr = classification_report(y_test, y_pred_randfor_bag)
print(cr)
print("\n")

#RANDOM FOREST WITH BOOSTING

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
boost = AdaBoostClassifier(base_estimator=classifier, n_estimators=10, random_state=42)
boost.fit(X_opt, y_train)

y_pred_randfor_boost = boost.predict(X_test)

print("----------------RANDOM FOREST WITH BOOSTING----------------")

accuracy = accuracy_score(y_test, y_pred_randfor_boost)
print('\nAccuracy:', accuracy)
print("\n")


cm = confusion_matrix(y_test, y_pred_randfor_boost)
print(cm)
print("\n")

cr = classification_report(y_test, y_pred_randfor_boost)
print(cr)
print("\n")

#LINEAR REGRESSION

lin_reg = LinearRegression()

lin_reg.fit(X_opt, y_train)
y_pred_lr = lin_reg.predict(X_test)

print("----------------LINEAR REGRESSION----------------")

mse = mean_squared_error(y_test, y_pred_lr)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_lr)
r2 = r2_score(y_test, y_pred_lr)
print("\nMean Squared Error :", mse)
print("\nRoot Mean Squared Error :", rmse)
print("\nMean Absolute Error :", mae)
print("\nR2 Score : ", r2)
print("\n")

#LINEAR REGRESSION WITH BAGGING

lin_reg = LinearRegression()
bag_reg = BaggingRegressor(base_estimator=lin_reg, n_estimators=10, random_state=42)

bag_reg.fit(X_opt, y_train)
y_pred_lr_bag = bag_reg.predict(X_test)

print("----------------LINEAR REGRESSION WITH BAGGING----------------")

mse = mean_squared_error(y_test, y_pred_lr_bag)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_lr_bag)
r2 = r2_score(y_test, y_pred_lr_bag)
print("\nMean Squared Error :", mse)
print("\nRoot Mean Squared Error :", rmse)
print("\nMean Absolute Error :", mae)
print("\nR2 Score : ", r2)
print("\n")

#LINEAR REGRESSION WITH BAGGING

linear_model = LinearRegression()
boost_reg = AdaBoostRegressor(base_estimator=linear_model, n_estimators=50, learning_rate=0.1, random_state=42)

boost_reg.fit(X_opt, y_train)
y_pred_lr_boost = boost_reg.predict(X_test)

print("----------------LINEAR REGRESSION WITH BOOSTING----------------")

mse = mean_squared_error(y_test, y_pred_lr_boost)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_lr_boost)
r2 = r2_score(y_test, y_pred_lr_boost)
print("\nMean Squared Error :", mse)
print("\nRoot Mean Squared Error :", rmse)
print("\nMean Absolute Error :", mae)
print("\nR2 Score : ", r2)
print("\n")

#LOGISTIC REGRESSION

classifier = LogisticRegression(random_state=42)
classifier.fit(X_opt, y_train)

y_pred_logrig = classifier.predict(X_test)

print("----------------LOGISTIC REGRESSION----------------")

accuracy = accuracy_score(y_test, y_pred_logrig)
print('\nAccuracy:', accuracy)
print("\n")


cm = confusion_matrix(y_test, y_pred_logrig)
print(cm)
print("\n")

cr = classification_report(y_test, y_pred_logrig)
print(cr)
print("\n")

#LOGISTIC REGRESSION WITH BAGGING

lr_model = LogisticRegression()
bagging_model = BaggingClassifier(base_estimator=lr_model, n_estimators=10, random_state=42)
bagging_model.fit(X_opt, y_train)

y_pred_logrig_bag = bagging_model.predict(X_test)

print("----------------LOGISTIC REGRESSION WITH BAGGING----------------")

accuracy = accuracy_score(y_test, y_pred_logrig_bag)
print('\nAccuracy:', accuracy)
print("\n")


cm = confusion_matrix(y_test, y_pred_logrig_bag)
print(cm)
print("\n")

cr = classification_report(y_test, y_pred_logrig_bag)
print(cr)
print("\n")

#LOGISTIC REGRESSION WITH BOOSTING

lr_model = LogisticRegression()
boost_clf = AdaBoostClassifier(base_estimator=lr_model, n_estimators=50, random_state=42)

boost_clf.fit(X_opt, y_train)
y_pred_logrig_boost = boost_clf.predict(X_test)

print("----------------LOGISTIC REGRESSION WITH BOOSTING----------------")

accuracy = accuracy_score(y_test, y_pred_logrig_boost)
print('\nAccuracy:', accuracy)
print("\n")


cm = confusion_matrix(y_test, y_pred_logrig_boost)
print(cm)
print("\n")

cr = classification_report(y_test, y_pred_logrig_boost)
print(cr)
print("\n")

#SVM

classifier = SVC(kernel='rbf', gamma='scale')
classifier.fit(X_opt, y_train)

y_pred_svm = classifier.predict(X_test)

print("----------------SVM----------------")

accuracy = accuracy_score(y_test, y_pred_svm)
print('\nAccuracy:', accuracy)
print("\n")


cm = confusion_matrix(y_test, y_pred_svm)
print(cm)
print("\n")

cr = classification_report(y_test, y_pred_svm)
print(cr)
print("\n")

#SVM WITH BAGGING

svm = SVC(kernel='rbf', gamma='scale')
classifier = BaggingClassifier(base_estimator=svm, n_estimators=10, random_state=42)
classifier.fit(X_opt, y_train)

y_pred_svm = classifier.predict(X_test)

print("----------------SVM WITH BAGGING----------------")

accuracy = accuracy_score(y_test, y_pred_svm)
print('\nAccuracy:', accuracy)
print("\n")


cm = confusion_matrix(y_test, y_pred_svm)
print(cm)
print("\n")

cr = classification_report(y_test, y_pred_svm)
print(cr)
print("\n")

#VOTING CLASSIFIER

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42)
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
lr_clf = LogisticRegression(random_state=42)
knn_clf = KNeighborsClassifier(n_neighbors=5)
dt_clf = DecisionTreeClassifier(random_state=42)

voting_clf = VotingClassifier(estimators=[('rf', rf_clf), ('ada', ada_clf), ('gb', gb_clf), ('lr', lr_clf), ('knn', knn_clf), ('dt', dt_clf)], voting='hard')
voting_clf.fit(X_opt, y_train)

y_pred_vote = voting_clf.predict(X_test)

print("----------------VOTING CLASSIFIER----------------")

accuracy = accuracy_score(y_test, y_pred_vote)
print('\nAccuracy:', accuracy)
print("\n")


cm = confusion_matrix(y_test, y_pred_vote)
print(cm)
print("\n")

cr = classification_report(y_test, y_pred_vote)
print(cr)
print("\n")

#STACKING

estimators = [('lr', LogisticRegression()),
              ('dt', DecisionTreeClassifier()),
              ('knn', KNeighborsClassifier())]

stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking.fit(X_opt, y_train)

y_pred_stack = stacking.predict(X_test)

print("----------------STACKING----------------")

accuracy = accuracy_score(y_test, y_pred_stack)
print('\nAccuracy:', accuracy)
print("\n")


cm = confusion_matrix(y_test, y_pred_stack)
print(cm)
print("\n")

cr = classification_report(y_test, y_pred_stack)
print(cr)
print("\n")
