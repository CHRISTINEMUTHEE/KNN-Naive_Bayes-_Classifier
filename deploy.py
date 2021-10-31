# Importing all the necessary libraries for my analysis

from matplotlib.colors import Normalize
import pandas as pd
import numpy as np


# Feature Engineering ,Preprocessing and Modeling

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# Function to diable warnings

import warnings
warnings.filterwarnings("ignore")


# Using the training features and the testing set to evaluate the best model on new data
clean_df_train=pd.read_csv('Clean_df_train.csv')

print(clean_df_train.columns)

print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")

# Checking the shape of my set and the datatypes
def Shape(data):
    data.shape
    types=data.dtypes
    print("My dataset has ",data.shape[0],"rows and ",data.shape[1]," columns of type : ")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(types)
Shape(clean_df_train)

# Defining my target variable
Y=clean_df_train['Survived'].values

X=clean_df_train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']].values

print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")

# Spliting the set into training and testing 

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

# Scaling the sets

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

# Baseline Model

def model (pred,target,test_size):

    # Splitting further into train test so that the model can learn Y using the training set as my test set doesnt have a target variable
    X_train,X_test,Y_train,Y_test=train_test_split(pred,target,test_size=test_size,random_state=42)

    # Instantiating KNN classifier
    knn=KNeighborsClassifier(n_neighbors=5)

    # Fitting my training set to the model
    knn.fit(X_train,Y_train)

    # Predicting from the knn classifier
    Y_real_pred=knn.predict(X_test)
    
    # Evaluating our model
    ac=accuracy_score(Y_test,Y_real_pred)
    cm=confusion_matrix(Y_test,Y_real_pred)
    cr=classification_report(Y_test,Y_real_pred)
    print(ac,'\n',cm,'\n',cr)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")

# Defining the function for my best model
def optimum(testing):

    # Tuning the metric and k neighbours to the best estimator to obtain the best model
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
    Y_best_pred=best_knn_model.predict(testing)

    # Evaluating our model
    ac_best=accuracy_score(Y_test,Y_best_pred)
    cm_best=confusion_matrix(Y_test,Y_best_pred)
    cr_best=classification_report(Y_test,Y_best_pred)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(ac_best,'\n',cm_best,'\n',cr_best)  
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")

# Instantiating the baseline model
model (X,Y,0.2)
# Calling the Optimum model
optimum(X_test)
