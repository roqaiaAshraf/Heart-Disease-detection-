import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV , cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle                                 
df = pd.read_csv('C:/Users/DELL/Desktop/Heart_Disease/heart.csv')
df
#1-age (Age of the patient in years)
#2-sex (Male/Female) (1/0)
#3-cp chest pain type ([typical angina, atypical angina, non-anginal, asymptomatic])(0~3)
#4-trestbps resting blood pressure (resting blood pressure (in mm Hg on admission to the hospital))
#5-chol (serum cholesterol in mg/dl)
#6-fbs (if fasting blood sugar > 120 mg/dl) (1/0)
#7-restecg (resting electrocardiographic results)-- Values: [normal, stt abnormality, lv hypertrophy] (0~2)  khrabet el 2lb
#8-thalach: maximum heart rate achieved
#9-exang: exercise-induced angina (True/ False) (1/0)
#10-oldpeak: ST depression induced by exercise relative to rest
#11-slope: the slope of the peak exercise ST segment (downsloping/flat/upsloping)
#12-ca: number of major vessels (0-3) colored by fluoroscopy
#13-thal: [normal; fixed defect; reversible defect] (1~3)
#14-target: yes or no (1/0)
df.info()
df.describe()

from sklearn.model_selection import GridSearchCV
# hyperparameters to tune
param_grid = {
    'n_estimators': [25, 50, 100, 150,200],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [3, 6, 9,11,13,15],
    'max_leaf_nodes': [3, 6, 9,11,13,15],
}
x=df.drop('target',axis='columns')
y=df.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

# creating RandomForest model
model = RandomForestClassifier()

# defining grid search object
grid = GridSearchCV(model, param_grid, cv=4)

# fitting the model for grid search
grid.fit(x_train, y_train)

# print best parameter after tuning
print(grid.best_params_)

accuracy_train = grid.best_estimator_.score(x_train, y_train)

accuracy = grid.best_estimator_.score(x_test, y_test)
print("Accuracy train:", accuracy_train)
print("Accuracy test:", accuracy)
# open file to write the model
with open('grid.pkl', 'wb') as file:
    # dump the model object into the file
    pickle.dump(grid, file) #put model in file

# print confirmation message
print("Trained model saved successfully!")


