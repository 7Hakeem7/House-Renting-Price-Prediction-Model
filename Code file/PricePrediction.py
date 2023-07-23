import plotly.express as px
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request , redirect , url_for 
import pickle
import numpy as np


#Removes the limit from the number of displayed columns and rows. 
# This is so I can see the entire dataframe when I print it 
pd.set_option("display.max_columns", None)

#pd.set_option('display.max_rows', None) 
pd.set_option("display.max_rows", 200)

#To build Linear model for statistical analysis and prediction
import statsmodels.stats.api as sms
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import (
    RandomForestRegressor, 
    AdaBoostRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

import warnings

warnings.filterwarnings("ignore")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 

#Reading the csv file
# Load the CSV file into a pandas DataFrame
# -*- coding: utf-8 -*-
df = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\House Renting Model\Housing.csv')

# View the first few rows of the DataFrame
print(df.head())

#Data Visualisation
df["mainroad"] = df["mainroad"].replace(["yes", "no"], [1, 0])
df["guestroom"] = df["guestroom"].replace(["yes", "no"], [1, 0])
df["basement"] = df["basement"].replace(["yes", "no"], [1, 0])
df["hotwaterheating"] = df["hotwaterheating"].replace(["yes", "no"], [1, 0])
df["airconditioning"] = df["airconditioning"].replace(["yes", "no"], [1, 0])
df["furnishingstatus"] = df["furnishingstatus"].replace(["furnished", "semi-furnished", "unfurnished"], [1, 1, 0])

#Checking coloumns
df.columns

#Outliers
def remove_outlier (col):
    sorted(col)
    q1, q3= np.quantile (col, [0.25, 0.75])
    iqr =  q3 - q1
    ll = q1 - 1.5 * iqr
    ul = q3 + 1.5* iqr
    return ll, ul

a = 1
plt.figure(figsize=(10, 50))
for i in df[    
      [
            'price',
            'area',
            'bedrooms',
            'bathrooms',
            'stories',
            'mainroad',
            'guestroom',
            'basement',
            'hotwaterheating',
            'airconditioning',
            'parking',
            'furnishingstatus'
      ]
]:
      if df[i].dtype != "object":
          plt.subplot(13, 2, a) 
          sns.distplot(df[i])
          # plt.title("Histogram for:" + i) 
          plt.subplot(13, 2, a + 1) 
          sns.boxplot(df[i])
          # plt.title("Boxplot for:" + i)
          a += 2

#depandable variable check
X1 = df.drop(["price"], axis=1) 
y1=df[["price"]]

X1.head()

# split the first dataset for price prediction into train and test
from sklearn.model_selection import train_test_split

X1_train, X1_test, y1_train, y1_test = train_test_split(
     X1, y1, test_size=0.3, random_state=42
)



#LINEAR REGRESSION
linearregression1 = LinearRegression() 
linearregression1.fit(X1_train, y1_train)


for idx, col_name in enumerate(X1_train.columns):
    print(
          "The coefficient for {} is {}".format(col_name, linearregression1.coef_[0][idx])
    )

print("The intercept for our model is {}".format(linearregression1.intercept_[0]))
print(X1_train.columns)

pred_train1 = linearregression1.predict(X1_train)
pred_test1 = linearregression1.predict(X1_test)

# R2 Squared:
print("R2 Squared for X1:")

lrscore_train1 = linearregression1.score (X1_train, y1_train) 
lrscore_test1 = linearregression1.score (X1_test, y1_test) 
print(lrscore_train1)
print(lrscore_test1)

data1 = y1_test.copy()
data1["pred1"] = pred_test1
data1["residual1"] = data1["price"] - data1["pred1"]
data1.head()

linearregression1 = LinearRegression()
linearregression1.fit(X1_train, y1_train)

filename = 'trainedModel.pkl'
pickle.dump(linearregression1,open(filename,'wb'))

linearregression1 = pickle.load(open('trainedModel.pkl','rb'))


#FLASK PART
app = Flask(__name__)

# Load the trained Linear Regression model from the .pkl file
# with open('trainedModel.pkl', 'rb') as file:
#     model = pickle.load(file)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         return render_template('predict')

#     return render_template('index.html')
@app.route("/")
def index():
  return render_template("index.html")

@app.route("/predict")
def inspect():
  return render_template("predict.html")

@app.route('/output')
def out():
    predicted_price = request.args.get('prediction')
    print("Predicted Price:", predicted_price)
    return render_template('output.html', output=predicted_price)





# ... Your existing code ...

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        stories = int(request.form['stories'])
        mainroad = int(request.form['mainroad'])
        guestroom = int(request.form['guestroom'])
        basement = int(request.form['basement'])
        hotwaterheating = int(request.form['hotwaterheating'])
        airconditioning = int(request.form['airconditioning'])
        parking = int(request.form['parking'])
        furnishingstatus = int(request.form['furnishingstatus'])

        # Prepare the input data for prediction
        input_data = np.array([[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning,parking, furnishingstatus]])

        # Load the trained Linear Regression model from the .pkl file
        with open('trainedModel.pkl', 'rb') as file:
            model = pickle.load(file)

        # Make prediction using the model
        # predicted_price = model.predict(input_data)[0]
        predicted_price = model.predict(input_data)[0][0]

        return redirect(url_for('out', prediction=predicted_price))
    else:
        return render_template('predict.html')




if __name__ == '__main__':
    app.run(debug=True)


