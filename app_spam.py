import streamlit as st

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
            st.metric(
                label="Probabilidad SPAM",
                value=f"{prob_spam:.2f}%"
            )
        with col2:
            st.metric(
                label="Probabilidad legítimo",
                value=f"{prob_no_spam:.2f}%"
            )

        st.subheader("Nivel de riesgo")
        st.progress(int(prob_spam) / 100)  # ✅ Corregido: dividido entre 100

        # MENSAJE EXTRA
        if prob_spam > 90:
            st.warning("⚠️ Riesgo extremadamente alto de spam")
        elif prob_spam > 70:
            st.info("ℹ️ El comentario contiene patrones típicos de spam")
        else:
            st.success("✔️ No se detectan señales fuertes de spam")


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

st.markdown(
    """
    <div class='footer'>
        Proyecto IA • Detector de Spam en YouTube • Streamlit App
    </div>
    """,
    unsafe_allow_html=True
)