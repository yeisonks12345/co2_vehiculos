import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('emisionesco2vehiculos.csv',sep = ';')

#separamos los datos

features = df.iloc[:,:-1]
target =df.iloc[:,-1]

#codificar variables categoricas en numericas con label


categorical_columns = ['Make', 'Model', 'Vehicle Class']
label_encoders = {}

for column in categorical_columns:
    le = LabelEncoder()
    features[column] = le.fit_transform(features[column])
    label_encoders[column] = le


#print(features.head(2))

# Codigo para Revertir las Columnas Numéricas a las Categóricas Originales
#for column in categorical_columns:
#    le = label_encoders[column]
#    features[column] = le.inverse_transform(features[column])

# aplicar modelos de SVM


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Paso 4: Entrenar el Modelo SVR
svr_model = SVR(kernel='linear')  # probar otros kernels como 'rbf', 'poly', etc.
svr_model.fit(X_train, y_train)

# Paso 5: Evaluar el Modelo
y_pred = svr_model.predict(X_test)


print("\nMean Squared Error:")
print(mean_squared_error(y_test, y_pred))

print("\nR^2 Score:")
print(r2_score(y_test, y_pred))