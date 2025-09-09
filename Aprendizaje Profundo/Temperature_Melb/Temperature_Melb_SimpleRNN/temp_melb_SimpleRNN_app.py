# Importando las librerías necesarias
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


#Funciones auxiliares (crear secuencias, predicción, etc.)
def predict_multistep(model, last_seq, scaler, days=7):
    """
    Predice los próximos días de temperatura mínima usando el modelo RNN.
    """
    preds = []
    current_seq = last_seq.reshape(1, -1, 1)  #Reshape para la entrada del modelo

    #Itera para predecir los próximos días
    for _ in range(days):
        pred = model.predict(current_seq, verbose=0)  #Predicción del modelo
        preds.append(pred[0, 0]) #Almacena la predicción
        current_seq = np.append(current_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)  #Actualiza la secuencia actual

    #Desnormaliza las predicciones
    preds = np.array(preds).reshape(-1, 1)  #Asegura que preds sea un array 2D
    return scaler.inverse_transform(preds)  #Desnormaliza las predicciones

#Cargamos el modelo y el escalador
scaler = joblib.load('D:\\Hacking\\Python\\AI_Learning\\Aprendizaje_Profundo\\Temperature_Melb_SimpleRNN\\melb_temp_scaler.pkl')
model_loaded = load_model('D:\\Hacking\\Python\\AI_Learning\\Aprendizaje_Profundo\\Temperature_Melb_SimpleRNN\\melb_temp_rnn_model.keras')

#Cargar datos originales
df = pd.read_csv('D:\\Hacking\\Python\\AI_Learning\\Aprendizaje_Profundo\\Temperature_Melb_SimpleRNN\\daily-minimum-temperatures-melb.csv')
df.rename(columns={'Daily minimum temperatures in Melbourne, Australia, 1981-1990': 'Temp'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df['Temp'] = pd.to_numeric(df['Temp'], errors='coerce')
df.dropna(inplace=True)

#Normalizar los datos
temp_scaled = scaler.transform(df[['Temp']].values)
steps = 30  #Número de pasos a predecir
last_seq = temp_scaled[-steps:]  #Últimos 30 días de datos normalizados

#Selector de fecha para la predicción
days = st.slider("Selecciona el número de días a predecir:", min_value=1, max_value=7, value=7)  #Número de días a predecir

#Interfaz de usuario con Streamlit
st.title(f"🌡️ Predicción de Temperatura en Melbourne ({days} día{'s' if days > 1 else ''})")

#Realizar la predicción
preds = predict_multistep(model_loaded, last_seq, scaler, days=days)
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days)

#Mostrar los resultados de la predicción
st.subheader("📈 Predicción para los próximos días")
# Mostrar las fechas y las temperaturas predichas
for date, temp in zip(future_dates, preds.flatten()):
    st.write(f"{date.strftime('%d/%m/%Y')}: {temp:.2f} °C")

#Graficar los resultados
st.subheader("📊 Gráfico de Predicción")

fig, ax = plt.subplots(figsize=(10, 4)) #Crear una figura y un eje para el gráfico
#Añadir los últimos 30 días al gráfico
ax.plot(df.index[-30:], scaler.inverse_transform(temp_scaled[-30:]), label='Últimos 30 dias', color='blue') 
ax.plot(future_dates, preds, label='Predicción', marker='o', color='red') #Añadir la predicción al gráfico
ax.set_xlabel('Fecha')
ax.set_ylabel('Temperatura (°C)')
ax.set_title('Predicción de Temperatura Mínima en Melbourne') 
ax.legend() #Añadir leyenda al gráfico

ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('/%d/%m')) #Formatear las fechas en el eje x
fig.autofmt_xdate()  #Formatear las fechas en el eje x
st.pyplot(fig)  #Mostrar el gráfico en Streamlit