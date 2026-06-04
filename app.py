import streamlit as st
import joblib

MODEL_PATH = "models/movie_sentiment_pipeline.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Review Sentiment Analysis")
st.write("This app predicts whether a movie review is positive or negative.")

review = st.text_area("Enter your movie review:", height=180)

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a movie review first.")
    else:
        model = load_model()
        prediction = model.predict([review])[0]

        if prediction == "positive":
            st.success("Positive Review 😊")
        else:
            st.error("Negative Review 😞")

st.markdown("---")
st.write("Built using Python, Scikit-learn, TF-IDF, Machine Learning, and Streamlit.")