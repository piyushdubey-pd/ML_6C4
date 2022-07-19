import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split as tts
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Iris.csv")
df.info()

X = df.iloc[:,[1,2,3,4]].values
y = df.iloc[:,5].values
X_train, X_test, y_train, y_test = tts(X,y,test_size=0.3)

import math,numpy as np
math.sqrt(len(df))

model = KNeighborsClassifier(n_neighbors = 13, metric = 'euclidean')
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print('Accuracy Metrics')
print(classification_report(y_test,y_pred))

import plotly.express as px
pred = model.predict(X)
fig1 = px.scatter(df, x="SepalWidthCm", y="SepalLengthCm", color=pred,
                 size='PetalLengthCm', hover_data=['PetalWidthCm'])
fig1.show()
cm = confusion_matrix(df['Species'], pred, labels=pred)
px.imshow(cm,text_auto=True,labels=dict(x="Predicted Label", y="True Label", color="No of classification"),
          x=pred,y=pred,title="Confusion Matrix",color_continuous_scale="aggrnyl")
