import pandas as pd
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
Data=pd.read_csv("Trust_Safety_Moderation.csv")
# print(Data.head())

# #mapping the valus(Label Incoding)
# print(Data["Action"].unique())
Data["Action"]=Data["Action"].map({
        "Allow":1,
        "Warn":2,
        "Restrict":3,
        "Ban":4
    })
# print(Data)
# print(Data["Severity"].unique())
Data["Severity"]=Data["Severity"].map({
    "No":0,
    "Low":1,
    "Medium":2,
    "High":3,
    "Critical":4
    })
#Training the data
Features=Data.drop(columns=["Action","User ID"])
Target=Data["Action"]
X_train,X_test,y_train,y_test=train_test_split(Features,Target,test_size=0.2,random_state=42,stratify=Target)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
model=LogisticRegression(max_iter=10000)
model.fit(X_train_scaled,y_train)
y_prob=model.predict_proba(X_test_scaled)
y_pred=model.predict(X_test_scaled)
cofficient=model.coef_
intercept=model.intercept_
print("The Probability values are",y_prob)
print("The Prediction calss is :", y_pred)
confussion=confusion_matrix(y_test,y_pred)
accurary=accuracy_score(y_test,y_pred)
classification_report=classification_report(y_test,y_pred)
print("The Accuracy of the Model is :",accurary)
# print(Data["Action"].value_counts())
print("The Confussion Matrics is : ","\n",confussion)
print("The Classification Report is ","\n",classification_report)
#Visualization 
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Allow","Warn","Restrict","Ban"]
)

plt.title("User Action Classification")
plt.show()

#Plotting the predictions
# plt.plot(
#     y_test.values,
#     marker='o',
#     label='Actual'
# )
# plt.plot(
#     y_pred,
#     marker='x',
#     label='Predicted'
# )
# plt.xlabel("Test Sample")
# plt.ylabel("Action Class")
# plt.title("Actual vs Predicted Actions")
# plt.legend()
# plt.grid(True)
# plt.show()

coef_df = pd.DataFrame(
    model.coef_,
    columns=Features.columns,
    index=model.classes_
)

print(coef_df)

#Plotting feature chart
# features = [
#     "All Features",
#     "No Timestamp",
#     "No Severity"
# ]
# accuracy = [
#     0.675,
#     0.675,
#     0.595
# ]
# plt.bar(features, accuracy)
# plt.ylabel("Accuracy")
# plt.title("Feature Impact Analysis")
# plt.show()

#plotting Feature Importance
# coef_df.plot(kind="bar", figsize=(10,6))

# plt.title("Logistic Regression Coefficients")
# plt.ylabel("Coefficient")
# plt.grid(True)
# plt.show()