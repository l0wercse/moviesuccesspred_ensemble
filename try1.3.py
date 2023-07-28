#Import all the necessary packages
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error,r2_score
import warnings
warnings.filterwarnings('ignore')


#Read the data of the csv onto a variable and add a new column
data=pd.read_csv('movie_metadata.csv')

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

#KNN WITH BAGGING
knn_model = KNeighborsRegressor(n_neighbors=5)

bagging_model = BaggingClassifier(base_estimator=knn_model, n_estimators=10, random_state=42)
bagging_model.fit(X_opt, y_train)
print("----------------KNN BAGGING----------------")


y_pred_knn_bag = bagging_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_knn_bag)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_knn_bag)
r2 = r2_score(y_test, y_pred_knn_bag)
print("\nMean Squared Error :", mse)
print("\nRoot Mean Squared Error :", rmse)
print("\nMean Absolute Error :", mae)
print("\nR2 Score : ", r2)

#KNN WITH BOOSTING
boosted_knn = GradientBoostingRegressor(knn_model, n_estimators=100, random_state=42)
boosted_knn.fit(X_opt, y_train)

print("----------------KNN BOOSTING----------------")

y_pred_knn_boost = boosted_knn.predict(X_test)

mse = mean_squared_error(y_test, y_pred_knn_boost)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_knn_boost)
r2 = r2_score(y_test, y_pred_knn_boost)
print("\nMean Squared Error :", mse)
print("\nRoot Mean Squared Error :", rmse)
print("\nMean Absolute Error :", mae)
print("\nR2 Score : ", r2)

#KNN WITH BEST K
k_values = range(1, 21)
best_k = 0
best_accuracy = 0
for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    accuracy = sum(cross_val_score(knn, X_train, y_train, cv=10, scoring="accuracy")) / 10
    if accuracy > best_accuracy:
        best_k = k
        best_accuracy = accuracy

knn = KNeighborsRegressor(n_neighbors=best_k)
knn.fit(X_opt, y_train)

print("----------------KNN WITH BEST K----------------")

y_pred_knn_bestk = knn.predict(X_test)

mse = mean_squared_error(y_test, y_pred_knn_bestk)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_knn_bestk)
r2 = r2_score(y_test, y_pred_knn_bestk)
print("\nMean Squared Error :", mse)
print("\nRoot Mean Squared Error :", rmse)
print("\nMean Absolute Error :", mae)
print("\nR2 Score : ", r2)

#DECISION TREES WITH BAGGING
tree_clf = DecisionTreeRegressor(random_state=42)
bag_clf = BaggingRegressor(tree_clf, n_estimators=10, random_state=42)

bag_clf.fit(X_opt, y_train)
y_pred_dtbag = bag_clf.predict(X_test)

print("----------------DECISION TREES WITH BAGGING----------------")

mse = mean_squared_error(y_test, y_pred_dtbag)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_dtbag)
r2 = r2_score(y_test, y_pred_dtbag)
print("\nMean Squared Error :", mse)
print("\nRoot Mean Squared Error :", rmse)
print("\nMean Absolute Error :", mae)
print("\nR2 Score : ", r2)

#DECISION TREES WITH BOOSTING
tree_clf = DecisionTreeRegressor(max_depth=1)
boost_clf = GradientBoostingRegressor(tree_clf, n_estimators=100, random_state=42)

boost_clf.fit(X_opt, y_train)
y_pred_dtboost = boost_clf.predict(X_test)

print("----------------DECISION TREES WITH BOOSTING----------------")

mse = mean_squared_error(y_test, y_pred_dtboost)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_dtboost)
r2 = r2_score(y_test, y_pred_dtboost)
print("\nMean Squared Error :", mse)
print("\nRoot Mean Squared Error :", rmse)
print("\nMean Absolute Error :", mae)
print("\nR2 Score : ", r2)


#RANDOM FOREST
classifier = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
classifier.fit(X_opt, y_train)

y_pred_randfor = classifier.predict(X_test)

print("----------------RANDOM FOREST----------------")

mse = mean_squared_error(y_test, y_pred_randfor)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_randfor)
r2 = r2_score(y_test, y_pred_randfor)
print("\nMean Squared Error :", mse)
print("\nRoot Mean Squared Error :", rmse)
print("\nMean Absolute Error :", mae)
print("\nR2 Score : ", r2)

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