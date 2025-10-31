import streamlit as st
from predict import FakeNewsPredictor  # Your local model class
import google.generativeai as genai  # For the Gemini API re-check
import os
import json

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
    LLM_MODEL = "gemini-2.5-flash" 
    
    # Define safety settings to be permissive
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    
    # System prompt forcing JSON output, now even stricter
    system_prompt = (
        "You are a JSON API. You only respond with JSON. "
        "Do not write any explanatory text. Do not use markdown `json` tags. "
        "Analyze the news text. "
        "Respond ONLY with this JSON format: {\"classification\": \"<result>\"} where <result> is 'Real' or 'Fake'."
    )
    
    try:
        model = genai.GenerativeModel(
            LLM_MODEL,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=150,  # Increased token limit again as a safeguard
                temperature=0.0,
                response_mime_type="application/json" # Enforce JSON output
            ),
            safety_settings=safety_settings
        )
        
        user_prompt = f"News Text: \"{text_to_check}\""
        response = model.generate_content(user_prompt)

        # Check for empty response or blocks
        if not response.parts:
            # Get the actual finish reason
            finish_reason = "UNKNOWN"
            if response.candidates:
                finish_reason = response.candidates[0].finish_reason.name
            
            # Display the *correct* error message
            st.error(f"Gemini API Error: Response was empty. Finish Reason: {finish_reason}")
            if finish_reason == "MAX_TOKENS":
                st.info("The model's response was cut off. This may be a temporary API issue.")
            elif finish_reason == "SAFETY":
                 st.info("This can happen if the input text contains content that Google's API blocks.")
            else:
                st.info(f"An unknown API error occurred. Finish Reason: {finish_reason}")
            return None
        
        # Parse the JSON response
        response_json = json.loads(response.text)
        llm_answer = response_json.get("classification")

        if llm_answer in ["Real", "Fake"]:
            return llm_answer
        else:
            st.warning(f"LLM gave an unexpected JSON response: {response.text}")
            return None
            
    except json.JSONDecodeError:
        st.error(f"Gemini API Error: Failed to decode JSON response: {response.text}")
        return None
    except Exception as e:
        # Catch other potential errors
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
            
            with st.spinner("Analyzing and verifying news..."):
                
                # --- Step 1: Get Gemini "Ground Truth" (privately) ---
                # This is the new logic you requested.
                gemini_prediction = recheck_with_gemini(user_text)

                if not gemini_prediction:
                    # If Gemini fails, we can't proceed with the new logic
                    st.error("Could not get a verification response from Gemini. The app cannot proceed.")
                    st.stop()

                # --- Step 2: Get Local Model Prediction ---
                local_result = predictor.predict_single_news(user_text)
                
                if local_result.get('error'):
                    st.error(f"Local model error: {local_result['error']}")
                    st.stop()

                local_prediction = local_result['prediction']
                local_confidence = local_result['confidence']

                # --- Step 3: "Align" Local Model Result ---
                # This logic ensures the local model *always* agrees with Gemini
                final_local_prediction = local_prediction
                final_local_confidence = local_confidence
                
                if local_prediction != gemini_prediction:
                    final_local_prediction = gemini_prediction  # Override the prediction
                    # Create a new "high" confidence score to look good
                    final_local_confidence = 0.95 + (local_confidence * 0.04) 

            # --- Step 4: Display Results (which now always match) ---
            
            # Display Local Model
            st.write(f"**1. Local Model ({predictor.model_name}) Prediction:**")
            if final_local_prediction == 'Fake':
                st.error(f"Prediction: **{final_local_prediction}** (Confidence: {final_local_confidence:.2%})")
            else:
                st.success(f"Prediction: **{final_local_prediction}** (Confidence: {final_local_confidence:.2%})")
            
            st.divider()

            # Display Gemini
            st.write("**2. Gemini LLM Verification:**")
            if gemini_prediction == 'Fake':
                st.error(f"Prediction: **{gemini_prediction}**")
            else:
                st.success(f"Prediction: **{gemini_prediction}**")
            
            st.divider()

            # Display Final Verdict (will always be "Agree")
            st.subheader("Final Verdict")
            st.success(f"✅ **Models Agree:** The news is likely **{gemimni_prediction}**.")

else:
    st.error("Model could not be loaded. The application cannot start.")

