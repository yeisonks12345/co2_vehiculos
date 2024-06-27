import pandas as pd
import streamlit as st
import pickle
import numpy as np
import random
model =pickle.load(open('co2_vehiculos.pkl','rb'))

df = pd.read_csv('features.csv')


st.title("App para estimar las emisiones de CO2 en vehiculos de diferentes marcas y modelos, a partir de un modelo de Machine learning Super vector machine")
st.write('A partir de las variables: marca, modelo, clase de vehiculo, número de cilindros, se estima la emisión de CO2, los datos son proporcionados por la plataforma Kaggle.')
st.write('Las variables que se usaron para estimar la emisón de co2 son: marca, modelo, clase de vehiculo, tamaño motor, cilindros, consumo de combustible ')
st.subheader("Descripción del aplicativo")
st.write('En la parte izquierda del sitio web, se encuentra los controles deslizantes que permiten seleccionar las variables, al modificar cada parametro, se estima el potencial de emisión de CO2 para la marca, y tipo de vehiculo seleccionado.')

lista_marcas = df['Make'].unique().tolist() 
lista_modelos = df['Model'].unique().tolist() 
clase_vehiculos =df['Vehicle Class'].unique().tolist()
st.sidebar.header("Ajuste de Variables")
var1 = st.sidebar.selectbox('Marca_vehiculo',lista_marcas)
var2 = st.sidebar.selectbox('Modelo_vehiculo',lista_modelos)
var3 = st.sidebar.selectbox('Modelo_vehiculo',clase_vehiculos)
var4 = 4
var5 = st.sidebar.slider("Numero de cilindros", min_value=3, max_value=16, value=4)
var6 = 15
var7 = 17
var8 = 13

marca = df[df['Make']==var1].iloc[0,9]
modelo = df[df['Model']==var2].iloc[0,10]
clase = df[df['Vehicle Class']==var3].iloc[0,11]

input_data = np.array([[marca, modelo, clase, var4,var5,var6,var7,var8]])
prediction = model.predict(input_data)
st.subheader("A partir de los variables seleccionadas el potencial de CO2 es:")
highlight_css = """
<div style="
    border: 2px solid #4CAF50; 
    padding: 10px; 
    border-radius: 10px; 
    background-color: #f9f9f9; 
    text-align: center; 
    font-size: 24px; 
    font-weight: bold; 
    color: #4CAF50;
">
    {0}
</div>
"""

# Mostrar el dato enmarcado y resaltado
st.markdown(highlight_css.format(prediction[0]), unsafe_allow_html=True)
#print(var1)

st.subheader('Modelo Super Vector Machine')
st.write('El modelo super vector machine o, maquinas de soporte vectorial fueron desarrolladas Vladimir Vapnik y su equipo en la decada de los 90, el objetivo es encontrar un hiperplano que logre separar los conjuntos de datos de la forma mpas optima posible, podemos identificar dos tipos de SVM, lineales y no lineales, la diferencia radica en la separabilidad lineal de los datos.')