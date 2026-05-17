
# IMPORTACIÓN DE LIBRERÍAS
import streamlit as st 
import pandas as pd 
import numpy as np
import plotly.express as px 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# CONFIGURACIÓN STREAMLIT
st.title("Clasificador de Diabetes - Grupo3")
st.write("Modelo de Machine Learning utilizando Random Forest")


# CARGA DE DATOS
df = pd.read_csv("diabetes.csv")


# EXPLORACIÓN DE DATOS - HEADER 1
st.header("1. Exploración de Datos y Tratamientos")

st.subheader("Primeras filas del dataset")
st.dataframe(df.head())

st.subheader("Estadísticas del Dataset")
st.dataframe(df.describe())

st.subheader("Información general")
info_df = pd.DataFrame({
    "Tipo de dato": df.dtypes.astype(str),
    "Valores nulos": df.isnull().sum().values
})
st.dataframe(info_df)

# DISTRIBUCIÓN VARIABLE OBJETIVO Y
st.subheader("Distribución de Diabetes")
fig = px.histogram( # Gráfico para contar datos
    df,
    x="Outcome",
    color="Outcome",
    text_auto=True # Mostrar contador
)

fig.update_layout(
    xaxis_title="Diabetes",
    yaxis_title="Cantidad de pacientes"
)

# Mostrar solo 0 y 1 en el eje X
fig.update_xaxes(
    tickmode='array',
    tickvals=[0, 1]
)

st.plotly_chart(fig, use_container_width=True) # Que se muestre en streamlit

st.write("El dataset no presenta valores nulos explícitos. Sin embargo, durante la exploración se detectaron múltiples valores igual a 0 en variables médicas donde dicho valor no es realista, por lo que estos podrían representar datos faltantes o errores de registro.")

st.subheader("Cantidad de valores 0")
columnas = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]
valores_cero = (df[columnas] == 0).sum()
st.dataframe(valores_cero)

# PREPROCESAMIENTO

# Tratamiento de Datos - Reemplazar valores 0 a NaN
df[columnas] = df[columnas].replace(0, np.nan)
# Rellenar con mediana
df.fillna(df.median(numeric_only=True), inplace=True)

st.write("Se realizó un tratamiento de datos debido a la presencia de valores igual a 0 en variables médicas donde dichos valores no son realistas. Estos fueron reemplazados por valores nulos y posteriormente imputados utilizando la mediana de cada variable.")

# Tabla de valores 0 post tratamiento
st.subheader("Cantidad de valores 0")
valores_cero = (df[columnas] == 0).sum()
st.dataframe(valores_cero)


X = df.drop("Outcome", axis=1) # Elimina columna Outcome del dataset
y = df["Outcome"] # Guarda columna objetivo

# Dividir datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# MODELO RANDOM FOREST
# Crea el modelo
modelo = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
# Entrena el modelo
modelo.fit(X_train, y_train)


# PREDICCIONES
y_pred = modelo.predict(X_test) # Prueba el modelo y predice Y


# MÉTRICAS - HEADER 2
# Respuestas reales vs predicciones 
st.header("2. Evaluación del Modelo")
col1, col2, col3 = st.columns(3)

# ¿Cuantas predicciones fueron correctas?
accuracy = accuracy_score(y_test, y_pred) 
col1.metric("Accuracy", f"{accuracy:.2f}")
# ¿Qué tantos positivos reales encontró el modelo?
recall = recall_score(y_test, y_pred)
col2.metric("Recall", f"{recall:.2f}")
# Precición y recall
f1 = f1_score(y_test, y_pred)
col3.metric("F1 Score", f"{f1:.2f}")


# MATRIZ DE CONFUSIÓN 
st.subheader("Matriz de Confusión")
datos_matriz = confusion_matrix(y_test, y_pred) # Compara valores reales y predicciones

fig2, ax = plt.subplots(figsize=(5,4)) # Crear figura

sns.heatmap(
    datos_matriz,
    annot=True,
    fmt="d", # Formato entero
    xticklabels=["No Diabetes", "Diabetes"],
    yticklabels=["No Diabetes", "Diabetes"],
    ax=ax
)

# Etiquetas
ax.set_xlabel("Predicción")
ax.set_ylabel("Valor Real")
st.pyplot(fig2)

st.write("La matriz de confusión muestra que el modelo logró clasificar correctamente gran parte de los pacientes. Se obtuvieron 78 verdaderos negativos y 37 verdaderos positivos. Sin embargo, existieron 21 falsos positivos y 18 falsos negativos, siendo estos últimos especialmente relevantes debido al riesgo médico asociado a no detectar pacientes con diabetes.")
st.write("El modelo presenta un desempeño aceptable para una primera aproximación, logrando una exactitud del 75%. Sin embargo, el recall de 0.67 indica que aún existen pacientes con diabetes que no son detectados correctamente. Debido a esto, el modelo podría mejorarse mediante limpieza de datos, tratamiento de valores atípicos, balanceo de clases o ajuste de hiperparámetros para aumentar su capacidad de detección.")

# PREDICCIÓN NUEVO PACIENTE - HEADER 3 INTERACTIVO
st.header("3. Predicción de Nuevo Paciente")
st.write("Ingrese los datos del paciente:")

# Fomulario - datos paciente
embarazo = st.number_input("Embarazo", min_value=0)
glucosa = st.number_input("Nivel de Glucosa", min_value=0)
presion_sanguinea = st.number_input("Presión Sanguinea", min_value=0)
espesor_piel = st.number_input("Espesor piel", min_value=0)
insulina = st.number_input("Insulina", min_value=0)
masa_corporal = st.number_input("BMI", min_value=0.0)
diabetes_historial = st.number_input("Riesgo historial de diabetes", min_value=0.0)
edad = st.number_input("Edad", min_value=0)


# BOTÓN DE PREDICCIÓN
if st.button("Predecir"):

    nuevo_paciente = pd.DataFrame([{
    "Pregnancies": embarazo,
    "Glucose": glucosa,
    "BloodPressure": presion_sanguinea,
    "SkinThickness": espesor_piel,
    "Insulin": insulina,
    "BMI": masa_corporal,
    "DiabetesPedigreeFunction": diabetes_historial,
    "Age": edad
}])

    prediccion = modelo.predict(nuevo_paciente)

    if prediccion[0] == 1:
        st.error("Warning: Se predice que el paciente TIENE diabetes")
    else:
        st.success("Todo ok: Se predice que el paciente NO tiene diabetes")
