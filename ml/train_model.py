import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
data = pd.read_csv('adhd_ratio_70.csv')
data.columns = data.columns.str.strip()
target = 'ADHD'
x = data.drop(target, axis=1)
y = data[target]
for col in x.columns:
    if (x[col].dtype=='object'):
        x[col] = x[col].fillna('None')
    else:
        x[col] = x[col].fillna(x[col].median())
label_encoders = {}

for col in x.columns:
    if (x[col].dtype == 'object'):
        le = LabelEncoder()
        x[col] = le.fit_transform(x[col])
        label_encoders[col] = le

xTrain, xValid, yTrain, yValid = train_test_split(x, y, test_size = 0.2, random_state = 42)
scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xValid = scaler.transform(xValid)
model = RandomForestClassifier(n_estimators = 1000,  random_state = 42, class_weight = 'balanced')
model.fit(xTrain, yTrain)
pred = model.predict(xValid)
print("Accuracy : ", accuracy_score(yValid, pred))
print(classification_report(yValid, pred))

joblib.dump(model, 'adhdModel.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'labelEncoders.pkl')
print("Pickle files saved")