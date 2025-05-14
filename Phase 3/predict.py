import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Data/cleaned_intrusion_traffic_data.csv') 

label_encoder = LabelEncoder()
df['Label_encoded'] = label_encoder.fit_transform(df['Label'])

features = df.drop(['Label', 'Label_encoded'], axis=1)
target = df['Label_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

df['Prediction'] = model.predict(features)
df['Prediction_Prob'] = model.predict_proba(features)[:, 1] 

df.to_csv('Data/enhanced_network_data.csv', index=False)

