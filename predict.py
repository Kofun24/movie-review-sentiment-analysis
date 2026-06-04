import joblib

MODEL_PATH = "models/movie_sentiment_pipeline.pkl"

model = joblib.load(MODEL_PATH)

print("Movie Review Sentiment Predictor")
print("Type 'exit' to stop")

while True:
    review = input("\nEnter a movie review: ")

    if review.lower() == "exit":
        print("Program stopped.")
        break

    prediction = model.predict([review])[0]

    print("Prediction:", prediction)