# Deploying the Naive bayes classifier
# Importing all the necessary libraries for my analysis
# EDA and Cleaning
from matplotlib.colors import Normalize
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Feature Engineering ,Preprocessing and Modeling
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder,Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
# Function to diable warnings
import warnings
warnings.filterwarnings("ignore")
# Using the training features and the testing set to evaluate the best model on new data
clean_df_train=pd.read_csv('Clean_df_train.csv')
print(clean_df_train.shape)
# Testing set
clean_df_test=pd.read_csv('Clean_df_test.csv')
print(clean_df_test.shape)
print(clean_df_train.columns) 
# Defining my target variable
Y=clean_df_train['Survived'].values
X=clean_df_train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']].values
X_validate=clean_df_test.values
# Spliting the set into training and testing 
print(X.shape)
print(X_validate.shape)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
# Scaling the sets
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
X_validate=sc.transform(X_validate)
# Instantiating the model using the params obtained for the best model
# Defining the function for my best model
def optimum(testing):
    # I will tune the metric and k neighbours to the best estimator to obtain the best model
    params={"metric":['minkowski','eucledian','cosine_similarity'],
        "n_neighbors": np.arange(1,13),
        "p":np.arange(1,3),
        "weights":['uniform','distance']}
    #Instantiating my model
    knn=KNeighborsClassifier()
    # Tuning the model to obtain the best parameters
    knn_search=GridSearchCV(estimator=knn,param_grid=params,cv=5,verbose=1)
    # Fitting my training set 
    knn_search.fit(X_train,Y_train)
    print("My best parameters are",knn_search.best_params_)
    # Obtaining the best estimator to make our prediction
    best_knn_model=knn_search.best_estimator_
    # Making our prediction using it
    Y_best_real_pred=best_knn_model.predict(testing)
    # Computing the baseline accuracy score
    ac_best=accuracy_score(Y_test,Y_best_real_pred)
    cm_best=confusion_matrix(Y_test,Y_best_real_pred)
    cr_best=classification_report(Y_test,Y_best_real_pred)
    print(ac_best,'\n',cm_best,'\n',cr_best)  
optimum(X_test)   
# testing the model on unseen data
print(X_train.shape)
print(X_test.shape)
# insert the values of X to obtain an outcome
X1,X2,X3,X4,X5,X6=[int(a) for a in input("Please insert six parameters ").split()]
knn=KNeighborsClassifier(n_neighbors=4,p=2)
X=[[X1,X2,X3,X4,X5,X6]]
Pred=knn.predict(X)
print(Pred)
# optimum(X)
