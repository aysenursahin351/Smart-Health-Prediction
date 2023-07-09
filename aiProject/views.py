from django.http import HttpResponseRedirect
from django.shortcuts import render
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

def prediction_graph(decision_tree_result, logistic_regression_result, random_forest_result, svm_result, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):
    x = ['', 'Decision Tree', 'Linear Regression', 'Random Forest Regression', 'Support Vector Result']
    y2 = [0, decision_tree_result, logistic_regression_result, random_forest_result, svm_result]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=y2, name='Predicted Outcome'))
    fig.update_layout(title='Prediction Model Comparison', xaxis_title='Model', yaxis_title='Probability')

    plot_div = fig.to_html(full_html=False)

    return plot_div

def getResultRandomForestRegression(X, y, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]
    prediction = model.predict(input_data)
    pred_formatted = "{:.2f}".format(prediction[0])

    return pred_formatted

def getResultSVM(X, y, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svm = SVR(kernel='rbf')
    svm.fit(X_scaled, y)

    X_test = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]
    X_test_scaled = scaler.transform(X_test)
    y_pred = svm.predict(X_test_scaled)
    pred_formatted = "{:.2f}".format(y_pred[0])
    return pred_formatted

def getResultLinearRegression(X, y, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    outcome_pred = model.predict([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
    outcome_pred_formatted = "{:.2f}".format(outcome_pred[0])

    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test, y_pred)

    return outcome_pred_formatted


def getResultDecisionTree(X, y, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeRegressor(random_state=1)
    model.fit(X, y)

    inputs = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]
    outcome_pred = model.predict(inputs)[0]

    y_pred = model.predict(X)
    accuracy = r2_score(y, y_pred)

    return outcome_pred


def getResultSupportVectorRegression(X, y, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svr = SVR(kernel='linear')
    svr.fit(X_scaled, y)

    X_test = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]
    X_test_scaled = scaler.transform(X_test)
    y_pred = svr.predict(X_test_scaled)
    pred_formatted = "{:.2f}".format(y_pred[0])

    return pred_formatted


def makePredict(request):
    if request.FILES:
        myfile = request.FILES['file']
        
        df = pd.read_csv(myfile)
        # Split data into X and y
        X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
        y = df['Outcome']
        pregnancies = float(request.POST['pregnancies'])
        glucose = float(request.POST['glucose'])
        blood_pressure = float(request.POST['blood_pressure'])
        skin_thickness = float(request.POST['skin_thickness'])
        insulin = float(request.POST['insulin'])
        bmi = float(request.POST['bmi'])
        diabetes_pedigree_function = float(request.POST['diabetes_pedigree_function'])
        age = float(request.POST['age'])
    if 'pregnancies' in request.POST and 'glucose' in request.POST and 'blood_pressure' in request.POST and 'skin_thickness'in request.POST and 'insulin' in request.POST and 'bmi' in request.POST and 'diabetes_pedigree_function'in request.POST and 'insulin' in request.POST and 'age' in request.POST:
    
        decision_tree_result = getResultDecisionTree(X, y, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)
        logistic_regression_result = getResultLinearRegression(X, y, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)
        random_forest_result = getResultRandomForestRegression(X, y, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)
        svm_result = getResultSVM(X, y, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)

        prediction_graph_returned = prediction_graph(decision_tree_result, logistic_regression_result, random_forest_result, svm_result, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)

        return render(request, 'calculation.html', {'randomForestResult': random_forest_result, 
                                                     'supportVectorResult': svm_result, 
                                                     'linearRegressionResult': logistic_regression_result, 
                                                     'decisionTreeResult': decision_tree_result, 
                                                     'pregnancies': pregnancies, 'glucose': glucose, 
                                                     'blood_pressure': blood_pressure, 'skin_thickness': skin_thickness, 
                                                     'insulin': insulin, 'bmi': bmi, 'diabetes_pedigree_function': diabetes_pedigree_function, 'age': age, 
                                                     'prediction_graph': prediction_graph_returned })
    else:
        return render(request, 'calculation.html')