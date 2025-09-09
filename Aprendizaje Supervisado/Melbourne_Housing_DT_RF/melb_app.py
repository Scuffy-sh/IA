#Importando las librerías necesarias
import streamlit as st
import pandas as pd
import joblib

#Cargando el modelo previamente entrenado
melb_model = joblib.load("D:\\Hacking\\Python\\AI_Learning\\Aprendizaje_Supervisado\\Melbourne_Housing\\melb_dt_model.pkl")

#Titulo de la aplicación
st.title("Predicción del Precio de Viviendas en Melbourne")

#Definiendo las características del pasajero
st.header("Ingrese los detalles de la vivienda")

#Creando los campos de entrada para las características de la vivienda
suburb = st.selectbox("Suburbio de la vivienda:", ['Abbotsford', 'Airport West', 'Albert Park', 'Altona', 'Altona North', 'Armadale', 'Ascot Vale', 'Ashburton', 'Balaclava', 'Bayswater', 'Bentleigh', 'Blackburn', 'Box Hill', 'Brighton', 'Brunswick', 'Camberwell', 'Carlton', 'Carnegie', 'Caulfield', 'Chadstone'])
rooms = st.number_input("Número de habitaciones:", min_value=1,)
type = st.selectbox("Tipo de vivienda (House, Unit):", ['House', 'Unit'])
method = st.selectbox("Método de venta (S, SP, VB):", ['S', 'SP', 'VB'])
seller = st.selectbox("Vendedor de la vivienda:", ['Biggin & Scott', 'Barry Plant', 'Hockingstuart', 'Jellis Craig', 'Ray White', 'McGrath', 'LJ Hooker', 'Nelson Alexander', 'Marshall White', 'Raine & Horne'])
distance = st.number_input("Distancia al centro de la ciudad (en km):", min_value=0.0, format="%.1f")
postal_code = st.number_input("Código postal:", min_value=3000, max_value=3999)
bedrooms = st.number_input("Número de dormitorios:", min_value=1)
bathrooms = st.number_input("Número de baños:", min_value=1)
carspaces = st.number_input("Número de espacios de estacionamiento:", min_value=0)
land_size = st.number_input("Tamaño del terreno (en m²):", min_value=0)
building_area = st.number_input("Área construida (en m²):", min_value=0)
year_built = st.number_input("Año de construcción:", min_value=1800, max_value=2025)
council_area = st.selectbox("Área del consejo:", ['Banyule', 'Bayside', 'Boroondara', 'Brimbank', 'Cardinia', 'Casey', 'Darebin', 'Frankston', 'Glen Eira', 'Greater Dandenong', 'Hobsons Bay', 'Hume', 'Kingston', 'Knox', 'Manningham', 'Maribyrnong', 'Maroondah', 'Melbourne', 'Melton', 'Monash'])
latitude = st.number_input("Latitud:", format="%.6f")
longitude = st.number_input("Longitud:", format="%.6f")
region = st.selectbox("Región de la vivienda:", ['Northern Metropolitan', 'Southern Metropolitan', 'Eastern Metropolitan', 'Western Metropolitan', 'South-Eastern Metropolitan'])
property_count = st.number_input("Número de propiedades en el área:", min_value=1)




if st.button("Predecir Precio"):
    #Creando un DataFrame con los datos del pasajero
    new_data = pd.DataFrame([{
        'Suburb': suburb,
        'Rooms': rooms,
        'Type': type,
        'Method': method,
        'Seller': seller,
        'Distance': distance,
        'PostalCode': postal_code,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Carspaces': carspaces,
        'LandSize': land_size,
        'BuildingArea': building_area,
        'YearBuilt': year_built,
        'CouncilArea': council_area,
        'Latitude': latitude,
        'Longitude': longitude,
        'Regionname': region,
        'PropertyCount': property_count
    }])
    
    #Codificando las variables categóricas
    new_data_encoded = pd.get_dummies(new_data)
    new_data_encoded = new_data_encoded.reindex(columns=melb_model.feature_names_in_, fill_value=0)
    
    #Realizando la predicción
    prediction = melb_model.predict(new_data_encoded)
    st.write("El precio estimado de la vivienda es de ", f"{prediction[0]:,.2f} dólares")
