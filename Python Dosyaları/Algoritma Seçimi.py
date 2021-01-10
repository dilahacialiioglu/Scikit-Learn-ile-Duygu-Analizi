#%% 
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

#%%

test_data = pd.read_csv("/content/test_2.csv")
train_data = pd.read_csv("/content/train_2.csv")
valid_data = pd.read_csv("/content/valid_2.csv")

data = pd.concat([test_data,train_data,valid_data] , axis = 0)

data.dropna(inplace = True)
#%%
tfidf=TfidfVectorizer(min_df=7, 
                      max_df=0.8,
                      strip_accents=None,
                      lowercase=False,
                      preprocessor=None,
                      use_idf=True,
                      norm='l2',
                      smooth_idf=True)

y=data.label.values
x_fit=tfidf.fit(data.text)
x = x_fit.transform(data.text)
#%% 
X_train,X_test,y_train,y_test=train_test_split(x,y,
                                               random_state=1,
                                               test_size=0.5,
                                               shuffle=False)
#%%
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()
param_dict = {
    "criterion":["gini" , "entropy"],
    "max_depth":range(1,10),
    "min_samples_split":range(1,10),
    "min_samples_leaf":range(1,5)}

grid = GridSearchCV(decision_tree,
                   param_grid = param_dict,
                   cv = 10,
                   verbose = 1,
                   n_jobs = -1)

grid.fit(X_train , y_train)

print(grid.best_params_)
print(grid.best_score_)

#%%
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(random_state = 42)
param_dict = { 
    'n_estimators': [1,10,50,100,200, 500,1000],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']}

grid_1 = GridSearchCV(random_forest,
                   param_grid = param_dict,
                   cv = 10,
                   verbose = 1,
                   n_jobs = -1)


grid_1.fit(X_train , y_train)

print(grid_1.best_params_)
print(grid_1.best_score_)

#%%
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]

grid_2= dict(solver=solvers,penalty=penalty,C=c_values)

grid_2 = GridSearchCV(estimator=log_reg, param_grid=grid_2, n_jobs=-1,cv = 10, scoring='accuracy',error_score=0)
grid_2.fit(X_train , y_train)

print_2(grid.best_params_)
print_2(grid.best_score_)

#%%
from sklearn.svm import SVC

svm = SVC(random_state = 42)
svm.fit(X_train,y_train)
print("Support Vector Machine Test Accuracy {}".format(svm.score(X_test , y_test)))