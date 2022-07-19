!pip install pgmpy

import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
from IPython.display import Image

df = pd.read_csv("data.csv")
df.info()

df.corr()["condition"]

import plotly.express as px
fig = px.histogram(df, x="cp", color='cp',pattern_shape="condition", title="Chest pain type VS Heart Disease")
fig.show()

fig = px.histogram(df, x="thal", color='thal',pattern_shape="condition", title="Thal VS Heart Disease",text_auto=True)
fig.show()

Model=BayesianNetwork([
                     ('age','trestbps'),('age','fbs'),
                     ('sex','trestbps'),
                     ('exang','trestbps'),
                     ('thal','condition'),
                     ('trestbps','condition'),
                     ('fbs','condition'),
                     ('condition','restecg'),('condition','thalach'),('condition','chol')
                     ])

def HeartDisease(x):
    prediction =  "No Heart Disease" if x[0] > x[1]  else "Heart Disease"
    print("\n Its predicted that the patient has " + prediction)
    
    

print('\n Learning CPD using Maximum likelihood estimators')
Model.fit(df,estimator=MaximumLikelihoodEstimator)

HeartDisease_infer = VariableElimination(Model)

print('\n 1. Probability of HeartDisease given Age=30')
q=HeartDisease_infer.query(variables=['condition'],evidence={'age':28})
print(q)
HeartDisease(q.values)

print('\n 2. Probability of HeartDisease given cholesterol=100')
q=HeartDisease_infer.query(variables=['condition'],evidence={'chol':100})
print(q)
HeartDisease(q.values)

print('\n 3. Probability of HeartDisease given cholesterol, thal')
q=HeartDisease_infer.query(variables=['condition'],evidence={'chol':282,'thal':2})
print(q)
HeartDisease(q.values)



#Burglary Case

EQ_model = BayesianNetwork(
    [
        ("Burglary", "Alarm"),
        ("Earthquake", "Alarm"),
        ("Alarm", "JohnCalls"),
        ("Alarm", "MaryCalls"),
    ]
)

cpd_B = TabularCPD(variable="Burglary", variable_card=2, values=[[0.001], [1 - 0.001]])

cpd_E = TabularCPD(variable="Earthquake", variable_card=2, values=[[0.002], [1 - 0.002]])

cpd_A = TabularCPD(
    variable="Alarm",
    variable_card=2,
    values=[[0.95, 0.94, 0.29, 0.001], [1 - 0.95, 1 - 0.94, 1 - 0.29, 1 - 0.001]],
    evidence=["Burglary", "Earthquake"],
    evidence_card=[2, 2],
)

cpd_J = TabularCPD(
    variable="JohnCalls",
    variable_card=2,
    values=[[0.90, 0.05], [0.10, 0.95]],
    evidence=["Alarm"],
    evidence_card=[2],
)

cpd_M = TabularCPD(
    variable="MaryCalls",
    variable_card=2,
    values=[[0.70, 0.01], [0.30,0.99]],
    evidence=["Alarm"],
    evidence_card=[2],
)

EQ_model.add_cpds(cpd_B, cpd_E, cpd_A, cpd_J, cpd_M)

EQ_model.check_model()

print(EQ_model.get_cpds("Burglary"))
print(EQ_model.get_cpds("Earthquake"))
print(EQ_model.get_cpds("Alarm"))
print(EQ_model.get_cpds("JohnCalls"))
print(EQ_model.get_cpds("MaryCalls"))
print("Nodes in the model:", EQ_model.nodes())

print(EQ_model.is_dconnected("Burglary", "Earthquake"))
print(EQ_model.is_dconnected("Burglary", "Earthquake", observed=["Alarm"]))

EQ_infer = VariableElimination(EQ_model)

q = EQ_infer.query(variables=["MaryCalls"], evidence={"Alarm":0})
print(q)
output = "Yes " if q.values[0] > q.values[1] else "No"
print('\n' +output)

print("Probability of burglary given both of them call\n")
q = EQ_infer.query(variables=["Burglary"], evidence={"MaryCalls":0,"JohnCalls":0})
print(q)
output = "Yes " if q.values[0] > q.values[1] else "No"
print('\n' +output)

print("Probability of Alarm given both of them call\n")
q = EQ_infer.query(variables=["Alarm"], evidence={"MaryCalls":0,"JohnCalls":0})
print(q)
output = "Yes " if q.values[0] > q.values[1] else "No"
print('\n' +output)

