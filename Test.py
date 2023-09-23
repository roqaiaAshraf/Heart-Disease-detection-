import pandas as pd 
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV , cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pickle         
with open('grid.pkl', 'rb') as file:
    # load the model object from the file
    model = pickle.load(file)
age= 52
sex=1  
cp=3 
trestbps=120 
chol=140 
fbs=1  
restecg=1 
thalach=140 
exang=1
oldpeak=1
slope=2  
ca=1 
thal=1   
# Make predictions on new data
input={'age' :age  ,'sex':sex ,'cp':cp  , 'trestbps':trestbps  , 
       'chol':chol ,'fbs':fbs  , 'restecg':restecg  , 'thalach':thalach , 
         'exang':exang  , 'oldpeak':oldpeak,  'slope':slope  , 'ca' :ca , 
         'thal':thal   
        }

predicted_class = model.predict([list(input.values())])

print(predicted_class)