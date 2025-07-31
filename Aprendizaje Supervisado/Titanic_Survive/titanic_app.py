#Importando las librerías necesarias
import streamlit as st
import pandas as pd
import joblib

#Cargando el modelo previamente entrenado
titanic_model = joblib.load("D:\\Hacking\\Python\\AI_Learning\\Aprendizaje_Supervisado\\Titanic\\titanic_rf_model.pkl")

#Titulo de la aplicación
st.title("Predicción de Supervivencia en el Titanic")

#Definiendo las características del pasajero
st.header("Ingrese los detalles del pasajero")

#Creando los campos de entrada para las características del pasajero
pclass = st.selectbox("Clase del pasajero (1, 2, 3):", [1, 2, 3])
sex = st.selectbox("Sexo del pasajero (male, female):", ['male', 'female'])
age = st.number_input("Edad del pasajero:", min_value=0)
sibsp = st.number_input("Número de hermanos/esposos a bordo:", min_value=0)
parch = st.number_input("Número de padres/hijos a bordo:", min_value=0)
fare = st.number_input("Tarifa del pasajero:", min_value=0.0, format="%.2f")
embarked = st.selectbox("Puerto de embarque (C, Q, S):", ['C', 'Q', 'S'])

if st.button("Predecir Supervivencia"):
    #Creando un DataFrame con los datos del pasajero
    new_passenger = pd.DataFrame([{
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked,
    }])
    
    #Codificando las variables categóricas
    new_passenger_encoded = pd.get_dummies(new_passenger)
    new_passenger_encoded = new_passenger_encoded.reindex(columns=titanic_model.feature_names_in_, fill_value=0)
    
    #Realizando la predicción
    prediction = titanic_model.predict(new_passenger_encoded)
    st.write("Resultado de la predicción:", "Sobrevivió" if prediction[0] == 1 else "No sobrevivió")
