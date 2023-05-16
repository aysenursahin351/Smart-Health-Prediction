from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import csv
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor 
import plotly.graph_objects as go

def index(request):
    return render(request, 'index.html')  

def process_file(request):
    return render(request, 'calculation.html')   

def train_model(model, X_train, y_train):
    """Modeli eğitir."""
    model.fit(X_train, y_train)

def prediction_graph(decision_tree_result, logistic_regression_result, random_forest_result, svm_result, high_bp, high_chol, chol_check):
       
    x = ['','Decision Tree', 'Lineer Regression', 'Random Forest Regression', 'Support Vector Result']
    y2 = [0, decision_tree_result, logistic_regression_result, random_forest_result, svm_result]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=y2, name='Predicted Diabetes'))
    fig.update_layout(title='Prediction Model Comparison', xaxis_title='Model', yaxis_title='Probability')

    plot_div = fig.to_html(full_html=False)
    
    return plot_div

def getResultRandomForestRegression(X, y, high_bp, high_chol, chol_check):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    input_data = np.array([[high_bp, high_chol, chol_check]])
    prediction = model.predict(input_data)
    pred_formatted = "{:.2f}".format(prediction[0])

    return pred_formatted


def getResultLinearRegression(X, y, high_bp, high_chol, chol_check):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    sales_pred = model.predict([[high_bp, high_chol, chol_check]])
    sales_pred_formatted = "{:.2f}".format(sales_pred[0])

    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test, y_pred)

    return sales_pred_formatted


def getResultDecisionTree(X, y, high_bp, high_chol, chol_check):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeRegressor(random_state=1)
    model.fit(X, y)

    inputs = [[high_bp, high_chol, chol_check]]
    sales_pred = model.predict(inputs)[0]

    y_pred = model.predict(X)
    accuracy = r2_score(y, y_pred)

    return sales_pred


def getResultSupportVectorRegression(X, y, high_bp, high_chol, chol_check):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svr = SVR(kernel='linear')
    svr.fit(X_scaled, y)

    X_test = [[high_bp, high_chol, chol_check]]
    X_test_scaled = scaler.transform(X_test)
    y_pred = svr.predict(X_test_scaled)
    pred_formatted = "{:.2f}".format(y_pred[0])

    return pred_formatted
def makeSalesPredict(request):
    if request.method == 'POST':
        if request.FILES:
            myfile = request.FILES['file']

           # Load data into a pandas DataFrame
        df = pd.read_csv(myfile)
        # Split data into x and y
        X = df[['HighBp', 'HighChol', 'CholCheck']]
        y = df['Diabetes']
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     

        if 'high_bp' in request.POST and 'high_chol' in request.POST and 'chol_check' in request.POST:
            # Get inputs from user
            high_bp = float(request.POST['high_bp'])
            high_chol = float(request.POST['high_chol'])
            chol_check = float(request.POST['chol_check']) 

            decision_tree_result = getResultDecisionTree(X, y, high_bp, high_chol, chol_check)
            logistic_regression_result = getResultLinearRegression(X, y, high_bp, high_chol, chol_check)
            random_forest_result = getResultRandomForestRegression(X, y, high_bp, high_chol, chol_check)
            svm_result = getResultSupportVectorRegression(X, y, high_bp, high_chol, chol_check) 
            
            prediction_graph_returned = prediction_graph(decision_tree_result, logistic_regression_result, random_forest_result, 
                                                       svm_result, high_bp, high_chol, chol_check) 

            return render(request, 'calculation.html', {'randomForestResult': random_forest_result, 
                                                       'supportVectorResult': svm_result, 
                                                       'linearRegressionResult': logistic_regression_result, 
                                                       'decisionTreeResult': decision_tree_result, 
                                                       'high_bp': high_bp, 'high_chol': high_chol, 
                                                       'prediction_graph': prediction_graph_returned }) 
            
        else:
                return render(request, 'calculation.html',context={"messageCode":2,"message":"Lütfen bir dosya yükleyiniz"})  #messageCode 1 başarılı 2 hatalı demek 
    else:
        return render(request, 'calculation.html')