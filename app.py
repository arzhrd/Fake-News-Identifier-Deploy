import streamlit as st
from predict import FakeNewsPredictor  # Your local model class
from openai import OpenAI            # Using OpenAI client for OpenRouter
import os

# --- Configuration ---

# Set the page title
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰 Fake News Detector (with OpenRouter/DeepSeek Verification)")
st.write("Enter text to check. The app will first use a local ML model, then verify the result with DeepSeek via OpenRouter.")

# --- Load API Key ---

# Try to get the key from Streamlit secrets (for deployment)
if 'OPENROUTER_API_KEY' in st.secrets:
    api_key = st.secrets['OPENROUTER_API_KEY']
else:
    # Fallback for local testing (set it as an environment variable)
    st.warning("OPENROUTER_API_KEY not found in Streamlit secrets. Falling back to environment variable.")
    api_key = os.environ.get("OPENROUTER_API_KEY")

# If no key is found, stop the app
if not api_key:
    st.error("🚨 OPENROUTER_API_KEY is not set. Please add it to your Streamlit secrets or environment variables.")
    st.stop()

# Initialize the OpenRouter client (using OpenAI's client)
client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

# --- Model Loading ---

@st.cache_resource
def load_model():
    """Loads the local FakeNewsPredictor model."""
    try:
        predictor = FakeNewsPredictor(model_path='fake_news_model.pkl')
        return predictor
    except Exception as e:
        st.error(f"Error loading local model: {e}")
        return None

# Load your local model
predictor = load_model()

# --- OpenRouter/DeepSeek Function ---

# Cache the API call to avoid re-running on the same text
@st.cache_data(ttl=3600)
def recheck_with_openrouter_deepseek(text_to_check):
    """
    Calls the OpenRouter API (using DeepSeek) to classify the text as 'Real' or 'Fake'.
    """
    # Use the free DeepSeek model via OpenRouter as per your reference
    LLM_MODEL = "deepseek/deepseek-chat-v3.1:free" 
    
    system_prompt = (
        "You are an expert fact-checker. Analyze the following news text. "
        "Classify it as 'Real' (trustworthy, factual) or 'Fake' (misinformation, clickbait, fabricated). "
        "Respond with only the single word: Real or Fake. Do not provide any explanation."
    )
    
    user_prompt = f"News Text: \"{text_to_check}\""
    
    try:
        response = client.chat.completions.create(
            extra_headers={
              # Optional headers for OpenRouter rankings
              # Make sure to replace this with your actual app URL once deployed!
              "HTTP-Referer": "https://fake-news-identifier-deploy-qw8mabrh2u7eif4uyxrdnw.streamlit.app/", 
              "X-Title": "Fake News Detector" 
            },
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=5,     # We only want one word
            temperature=0.0   # We want a deterministic, confident answer
        )
        llm_answer = response.choices[0].message.content.strip()
        
        # Clean the answer just in case the LLM adds punctuation
        if "real" in llm_answer.lower():
            return "Real"
        elif "fake" in llm_answer.lower():
            return "Fake"
        else:
            # The LLM failed to follow instructions
            st.warning(f"LLM gave an unexpected response: {llm_answer}")
            return None
            
    except Exception as e:
        st.error(f"OpenRouter API Error: {str(e)}")
        return None

# --- Streamlit UI ---

if predictor:
    user_text = st.text_area("Enter News Text:", "", height=200)

    if st.button("Analyze and Verify News"):
        if not user_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            st.subheader("Analysis Results")
            
            # --- Step 1: Get Local Model Prediction ---
            with st.spinner("Analyzing with local model..."):
                local_result = predictor.predict_single_news(user_text)
            
            if local_result.get('error'):
                st.error(f"Local model error: {local_result['error']}")
                st.stop()
            
            local_prediction = local_result['prediction']
            local_confidence = local_result['confidence']
            
            st.write(f"**1. Local Model ({predictor.model_name}) Prediction:**")
            if local_prediction == 'Fake':
                st.error(f"Prediction: **{local_prediction}** (Confidence: {local_confidence:.2%})")
            else:
                st.success(f"Prediction: **{local_prediction}** (Confidence: {local_confidence:.2%})")
            
            st.divider()

            # --- Step 2: Recheck with OpenRouter/DeepSeek ---
            st.write("**2. OpenRouter (DeepSeek) LLM Verification:**")
            with st.spinner("Verifying with OpenRouter/DeepSeek..."):
                llm_prediction = recheck_with_openrouter_deepseek(text_to_check) # Call the updated function

            if llm_prediction:
                if llm_prediction == 'Fake':
                    st.error(f"Prediction: **{llm_prediction}**")
                else:
                    st.success(f"Prediction: **{llm_prediction}**")
                
                st.divider()

                # --- Step 3: Final Verdict ---
                st.subheader("Final Verdict")
                if local_prediction == llm_prediction:
                    st.success(f"✅ **Models Agree:** The news is likely **{llm_prediction}**.")
                else:
                    st.warning(f"⚠️ **Prediction Conflict!**")
                    st.write(f"- Your local model said: **{local_prediction}**")
                    st.write(f"- OpenRouter/DeepSeek LLM said: **{llm_prediction}**")
                    st.info(f"The LLM's answer (**{llm_prediction}**) is often more reliable. Please use this as the final answer.")
            
            else:
                st.error("Could not get a verification response from OpenRouter.")

else:
    st.error("Model could not be loaded. The application cannot start.")
