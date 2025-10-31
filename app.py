import streamlit as st
from predict import FakeNewsPredictor # Import your class from predict.py

# Set the page title and a simple introduction
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰 Fake News Detector")
st.write("Enter a news headline or article text below to check if it's real or fake.")

# Caching the model
# @st.cache_resource ensures the model is loaded only ONCE,
# making your app much faster.
@st.cache_resource
def load_model():
    """Loads the FakeNewsPredictor model."""
    try:
        predictor = FakeNewsPredictor(model_path='fake_news_model.pkl')
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the predictor
predictor = load_model()

if predictor:
    # Create a text area for user input
    user_text = st.text_area("Enter News Text:", "", height=200)

    # Create a button to trigger the prediction
    if st.button("Analyze News"):
        if not user_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            # Perform prediction
            with st.spinner("Analyzing..."):
                result = predictor.predict_single_news(user_text)
            
            if result.get('error'):
                st.error(f"Error: {result['error']}")
            else:
                # Display the results
                st.subheader("Analysis Result")
                
                prediction = result['prediction']
                confidence = result['confidence']
                
                if prediction == 'Fake':
                    st.error(f"Prediction: **{prediction}**")
                else:
                    st.success(f"Prediction: **{prediction}**")

                # Display confidence with a metric and a progress bar
                st.metric(label="Confidence Score", value=f"{confidence:.2%}")
                
                # Show probabilities
                st.subheader("Probabilities")
                probs = result['probabilities']
                st.write(f"**Real News:** {probs['Real']:.2%}")
                st.write(f"**Fake News:** {probs['Fake']:.2%}")

                # Optional: Show the model used
                st.info(f"Model Used: {result['model_used']}")

else:
    st.error("Model could not be loaded. The application cannot start.")
