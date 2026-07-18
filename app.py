import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Credit Scoring App v2", layout="centered")

# Cargar modelo, scaler, columnas y umbral
modelo = joblib.load('modelo_riesgo_rf_v2.pkl')
scaler = joblib.load('scaler_v2.pkl')
columnas_modelo = joblib.load('columnas_modelo_v2.pkl')
umbral = joblib.load('umbral_v2.pkl')

st.title("Simulador de Riesgo Crediticio")
st.markdown("Modelo entrenado con 6 variables: duración, importe, historial crediticio, estado de cuenta, ahorros y antigüedad laboral.")

col1, col2 = st.columns(2)

opciones_savings = {
    'Sin ahorros previos': 'no known savings',
    'Inferior a 100€': '<100',
    'Entre 100 y 500€': '100<=X<500',
    'Entre 500 y 1000€': '500<=X<1000',
    'Mayor o igual a 1000€': '>=1000'    
}

opciones_checkings = {
    'Entre 0 y 200':'0<=X<200',
    'Inferior a 0': '<0',
    'Superior o igual a 200': '>=200',
    'Sin cuenta corriente': 'no checking'
}

opciones_history = {
    'Historial crítico o con otros créditos activos':'critical/other existing credit',
    'Créditos anteriores pagados correctamente':'existing paid',
    'Retraso en pagos anteriores':'delayed previously',
    'Sin créditos previos o todos liquidados':'no credits/all paid',
    'Todos los créditos anteriores':'all paid'
}

opciones_employments = {
    'Mayor o igual a 7':'>=7',
    'Entre 1 y 4':'1<=X<4',
    'Entre 4 y 7': '4<=X<7',
    'Desempleado': 'unemployed',
    'Menor a 1':'<1'
}
with col1:
    duration = st.number_input("Duración del préstamo (meses)", min_value=6, max_value=48, value=24)
    credit_amount = st.number_input("Importe solicitado (€)", min_value=100, max_value=10000, value=3000)

with col2:
    etiqueta_checking = st.selectbox("Estado de la cuenta corriente", list(opciones_checkings.keys()))
    checking_status = opciones_checkings[etiqueta_checking]
    
    etiqueta_savings = st.selectbox("Nivel de ahorros", list(opciones_savings.keys()))
    savings_status = opciones_savings[etiqueta_savings]

etiqueta_history = st.selectbox("Historial crediticio", list(opciones_history.keys()))
credit_history = opciones_history[etiqueta_history]

etiqueta_employment = st.selectbox("Antigüedad en el empleo actual", list(opciones_employments.keys()))
employment = opciones_employments[etiqueta_employment]

if st.button("Evaluar Riesgo", use_container_width=True):
    datos_cliente = {
        'duration': duration,
        'credit_amount': credit_amount,
        'checking_status': checking_status,
        'credit_history': credit_history,
        'savings_status': savings_status,
        'employment': employment
    }
    
    df_cliente = pd.DataFrame([datos_cliente]) 
    df_encoded = pd.get_dummies(df_cliente)  
    df_final = df_encoded.reindex(columns=columnas_modelo, fill_value=0)
    
    cols_numericas = ['duration', 'credit_amount']
    df_final[cols_numericas] = scaler.transform(df_final[cols_numericas])
    probabilidad = modelo.predict_proba(df_final)[0][0]
    
    if probabilidad >= umbral:
        st.error(f"PRÉSTAMO DENEGADO. Probabilidad de impago: {probabilidad*100:.1f}%")
    else:
        st.success(f"PRÉSTAMO APROBADO. Probabilidad de impago: {probabilidad*100:.1f}%")   
        