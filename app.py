import streamlit as st
from predict import FakeNewsPredictor  # Your local model class
import google.generativeai as genai  # For the Gemini API re-check
import os

# --- Configuration ---

# Set the page title
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰 Fake News Detector (with Gemini Verification)")
st.write("Enter text to check. The app will first use a local ML model, then verify the result with the Gemini LLM.")

# --- Load API Key ---

# Try to get the key from Streamlit secrets (for deployment)
if 'GEMINI_API_KEY' in st.secrets:
    api_key = st.secrets['GEMINI_API_KEY']
else:
    # Fallback for local testing (set it as an environment variable)
    st.warning("GEMINI_API_KEY not found in Streamlit secrets. Falling back to environment variable.")
    api_key = os.environ.get("GEMINI_API_KEY")

# If no key is found, stop the app
if not api_key:
    st.error("🚨 GEMINI_API_KEY is not set. Please add it to your Streamlit secrets or environment variables.")
    st.stop()

# Configure the Gemini client
genai.configure(api_key=api_key)

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

# --- Gemini Function ---

# Cache the API call to avoid re-running on the same text
@st.cache_data(ttl=3600)
def recheck_with_gemini(text_to_check):
    """
    Calls the Gemini API to classify the text as 'Real' or 'Fake'.
    """
    # Use a fast and powerful Gemini model"
    
    LLM_MODEL = "gemini-2.5-flash" 
    
    # We configure the model with safety settings and the system prompt
    # Gemini's 'system_instruction' is similar to OpenAI's 'system' role
    model = genai.GenerativeModel(
        LLM_MODEL,
        system_instruction=(
            "You are an expert fact-checker. Analyze the following news text. "
            "Classify it as 'Real' (trustworthy, factual) or 'Fake' (misinformation, clickbait, fabricated). "
            "Respond with only the single word: Real or Fake. Do not provide any explanation."
        ),
        generation_config=genai.GenerationConfig(
            max_output_tokens=5,  # We only want one word
            temperature=0.0       # We want a deterministic, confident answer
        )
    )
    
    user_prompt = f"News Text: \"{text_to_check}\""
    
    try:
        response = model.generate_content(user_prompt)
        llm_answer = response.text.strip()
        
        # Clean the answer just in case
        if "real" in llm_answer.lower():
            return "Real"
        elif "fake" in llm_answer.lower():
            return "Fake"
        else:
            # The LLM failed to follow instructions
            st.warning(f"LLM gave an unexpected response: {llm_answer}")
            return None
            
    except Exception as e:
        st.error(f"Gemini API Error: {str(e)}")
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

            # --- Step 2: Recheck with Gemini ---
            st.write("**2. Gemini LLM Verification:**")
            with st.spinner("Verifying with Gemini..."):
                llm_prediction = recheck_with_gemini(user_text) # Call the new Gemini function

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
                    st.write(f"- Gemini LLM said: **{llm_prediction}**")
                    st.info(f"The LLM's answer (**{llm_prediction}**) is often more reliable. Please use this as the final answer.")
            
            else:
                st.error("Could not get a verification response from Gemini.")

else:
    st.error("Model could not be loaded. The application cannot start.")
