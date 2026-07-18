import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuración básica de la página
st.set_page_config(page_title="Credit Scoring App", page_icon="🏦", layout="centered")

# Cargar el modelo y las columnas (con caché para agilizarlo)
@st.cache_resource
def cargar_modelo():
    modelo = joblib.load('modelo_riesgo_rf.pkl')
    columnas = joblib.load('columnas_modelo.pkl')
    return modelo, columnas

modelo_rf, columnas_modelo = cargar_modelo()

# Interfaz de Usuario (UI)
st.title("🏦Simulador de Riesgo Crediticio")
st.markdown("Introduce los datos del cliente para evaluar la probabilidad de impago mediante Inteligencia Artificial.")

# Creación dos columnas visuales
col1, col2 = st.columns(2)

with col1:
    importe = st.number_input("Importe Solicitado (€)", min_value=100, max_value=50000, value=5000)
    edad = st.number_input("Edad del Solicitante", min_value=18, max_value=100, value=35)
    
with col2:
    duracion = st.number_input("Duración del préstamo (Meses)",min_value=6, max_value=72, value=24)
    ratio_endeudamiento = st.selectbox("Ratio de Endeudamiento (1=Bajo, 4=Alto)", [1, 2, 3, 4], index=1)


st.markdown("---")
st.subheader("Política de Riesgos")

# Añadimos un slider para que el usuario elija qué tan estricto quiere ser
umbral_tolerancia = st.slider("Tolerancia máxima al riesgo de impago", min_value=0.10,max_value=0.90, value=0.30, step=0.05, help="Por encima de este porcentaje, el préstamo se deniega automáticamente.") 

# Configuración botón y lógica matemática
if st.button("Evaluar Riesgo", use_container_width=True):
    # Crearemos un diccionario con ceros para TODAS las columnas
    # Así la IA recibe el tamaño exacto de datos que espera.
    datos_entrada = {col: 0 for col in columnas_modelo} 
    
    # Actualizamos solo las variables que el usuario ha tocado en la web
    datos_entrada['Importe_Solicitado'] = importe
    datos_entrada['Edad'] = edad
    datos_entrada['Duracion_Meses']= duracion
    datos_entrada['Ratio_Endeudamiento'] = ratio_endeudamiento
     
    # Convertimos a formato tabla (Pandas)
    df_entrada = pd.DataFrame([datos_entrada])
    
    # Hacemos la predicción
    probabilidad_moroso = modelo_rf.predict_proba(df_entrada)[0][1] # Extraemos el % de ser Moroso (clase 1)
    
    # Aplicamos nuestra política estricta de negocio (Umbral 30%)
    st.markdown("---")
    if probabilidad_moroso >= umbral_tolerancia:
        st.error(f" PRÉSTAMO DENEGADO. Alto Riesgo Detectado.")
        st.warning(f"Probabilidad de impago: {probabilidad_moroso * 100:.1f}% (Supera el límite del 30%)")
        
        # --- EXPLICABILIDAD (EL POR QUÉ) ---
        with st.expander(" Motivos de la decisión del modelo:"):
            st.write("El algoritmo penaliza esta solicitud basándose en sus variables de mayor peso:")
            st.write(f"** Importe ({importe}€): ** La magnitud de la exposición al riesgo es el factor principal.")
            st.write(f"- **Edad ({edad} años):** Correlacionado estadísticamente con la madurez y estabilidad laboral histórica.")
            st.write(f"- **Duración ({duracion} meses):** Plazos prolongados aumentan la probabilidad de imprevistos económicos.")
            st.info("Sugerencia: Intente reducir el importe solicitado o acortar el plazo en meses para mejorar el *Credit Score*.")
    else:
        st.success(f"PRÉSTAMO APROBADO. Cliente Sólido.")
        st.info(f"Probabilidad de impago: {probabilidad_moroso * 100:.1f}% (Dentro de los límites)")