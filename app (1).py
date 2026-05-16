import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer

# ── Configuración de la página ──────────────────────────────────────────
st.set_page_config(
    page_title="Detector de Spam YouTube",
    page_icon="🎬",
    layout="centered"
)

# ── Carga del modelo y tokenizer ────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    model = load_model('modelo_spam.keras')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = cargar_modelo()

# ── Función de predicción ────────────────────────────────────────────────
def predecir(comentario):
    matriz = tokenizer.texts_to_matrix([comentario], mode='tfidf')
    matriz = matriz / np.amax(np.absolute(matriz))
    matriz = matriz - np.mean(matriz)
    prediccion = model.predict(matriz, verbose=0)
    prob_spam    = prediccion[0][1] * 100
    prob_no_spam = prediccion[0][0] * 100
    es_spam = prob_spam > prob_no_spam
    return es_spam, prob_spam, prob_no_spam

# ── Interfaz ─────────────────────────────────────────────────────────────
st.title("🎬 Detector de Spam en comentarios de YouTube")
st.markdown("Introduce un comentario y la red neuronal determinará si es **spam** o no.")

st.divider()

comentario = st.text_area(
    "💬 Comentario a analizar",
    placeholder="Escribe aquí el comentario...",
    height=120
)

if st.button("🔍 Analizar", use_container_width=True):
    if comentario.strip() == "":
        st.warning("Por favor, escribe un comentario antes de analizar.")
    else:
        es_spam, prob_spam, prob_no_spam = predecir(comentario)

        st.divider()

        if es_spam:
            st.error("🚨 SPAM detectado")
        else:
            st.success("✅ Comentario legítimo")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probabilidad SPAM", f"{prob_spam:.1f}%")
        with col2:
            st.metric("Probabilidad legítimo", f"{prob_no_spam:.1f}%")

        st.progress(int(prob_spam))

st.divider()
st.caption("Red neuronal entrenada con comentarios de YouTube — Proyecto 06")