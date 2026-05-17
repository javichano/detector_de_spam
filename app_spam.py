import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────────────────────
# CARGA DEL MODELO Y TOKENIZER
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    modelo = load_model("modelo_spam.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return modelo, tokenizer

modelo, tokenizer = cargar_modelo()

# ─────────────────────────────────────────────────────────────
# FUNCIÓN PREDECIR
# ─────────────────────────────────────────────────────────────
def predecir(comentario):
    # Transformar el texto igual que en el entrenamiento
    matriz = tokenizer.texts_to_matrix([comentario], mode='tfidf')
    matriz = matriz / np.amax(np.absolute(matriz))
    matriz = matriz - np.mean(matriz)

    # Predicción
    prediccion = modelo.predict(matriz, verbose=0)
    prob_no_spam = prediccion[0][0] * 100
    prob_spam    = prediccion[0][1] * 100
    es_spam      = prob_spam > prob_no_spam

    return es_spam, prob_spam, prob_no_spam

# ─────────────────────────────────────────────────────────────
# INTERFAZ
# ─────────────────────────────────────────────────────────────
comentario = st.text_area("Escribe un comentario de YouTube para analizar:")

if st.button("Analizar"):
    if comentario.strip() == "":
        st.warning("⚠️ Escribe un comentario antes de analizar.")
    else:
        with st.spinner("Analizando comentario..."):
            es_spam, prob_spam, prob_no_spam = predecir(comentario)
        st.divider()

        # RESULTADO PRINCIPAL
        if es_spam:
            st.error("🚨 El modelo detecta que el comentario es SPAM")
        else:
            st.success("✅ El comentario parece legítimo")

        # PROBABILIDADES
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Probabilidad SPAM",      value=f"{prob_spam:.2f}%")
        with col2:
            st.metric(label="Probabilidad legítimo",  value=f"{prob_no_spam:.2f}%")

        st.subheader("Nivel de riesgo")
        st.progress(int(prob_spam) / 100)

        # MENSAJE EXTRA
        if prob_spam > 90:
            st.warning("⚠️ Riesgo extremadamente alto de spam")
        elif prob_spam > 70:
            st.info("ℹ️ El comentario contiene patrones típicos de spam")
        else:
            st.success("✔️ No se detectan señales fuertes de spam")

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📌 Información")
    st.write(
        """
        Esta aplicación utiliza una red neuronal entrenada
        para detectar comentarios spam en YouTube.
        """
    )
    st.write("---")
    st.write("### Tecnologías")
    st.write("- Streamlit")
    st.write("- TensorFlow")
    st.write("- Scikit-learn")
    st.write("- NLP")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class='footer'>
        Proyecto IA • Detector de Spam en YouTube • Streamlit App
    </div>
    """,
    unsafe_allow_html=True
)
