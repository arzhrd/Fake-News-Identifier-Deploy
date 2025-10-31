📰 Fake News Detector (with LLM Verification)

This Streamlit application provides a two-step process for detecting fake news. It first uses a locally-trained machine learning model for an initial prediction and then verifies the result using the powerful Google Gemini API for a more reliable final verdict.

🚀 Overview

The goal of this project is to create a robust and reliable fake news detector. It combines the speed of a specialized, locally-trained model (fake_news_model.pkl) with the advanced reasoning and real-world knowledge of a large language model (LLM) like Google's Gemini.

✨ Features

Simple Interface: A clean and simple UI built with Streamlit to paste and analyze news text.

Local Model Prediction: Get an instant classification (Real/Fake) and confidence score from a pre-trained scikit-learn model.

LLM Verification: The app automatically sends the text to the Gemini API (gemini-2.5-flash) for a second opinion.

Final Verdict: The app compares both predictions:

If they agree, it gives a high-confidence "Models Agree" verdict.

If they conflict, it highlights the discrepancy and suggests that the LLM's answer is likely more reliable.

🛠️ How to Set Up and Run

You can run this project locally on your machine or deploy it to Streamlit Cloud.

1. Project Files

Your project directory should have the following structure:

/your-project-folder
│
├── app.py              # The Streamlit web application
├── predict.py          # Class for loading and using the local model
├── fake_news_model.pkl   # Your trained ML model file
├── requirements.txt    # Python dependencies
└── README.md           # This file


2. Dependencies

The requirements.txt file must contain all necessary packages:

streamlit
pandas
nltk
scikit-learn
google-generativeai


3. Running Locally

Clone the Repository:

git clone [https://github.com/your-github-username/your-repo-name.git](https://github.com/your-github-username/your-repo-name.git)
cd your-repo-name


Create a Virtual Environment (Recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Requirements:

pip install -r requirements.txt


Set Your API Key:
You must set your Gemini API key as an environment variable.

On macOS/Linux:

export GEMINI_API_KEY="your-actual-gemini-api-key-goes-here"


On Windows (Command Prompt):

set GEMINI_API_KEY="your-actual-gemini-api-key-goes-here"


Run the App:

streamlit run app.py


Your app will open in your browser at http://localhost:8501.

4. Deploying to Streamlit Cloud

Push to GitHub:

Create a new repository on GitHub.

Push your project files (app.py, predict.py, requirements.txt, fake_news_model.pkl).

Note on Large Files: If your fake_news_model.pkl file is larger than 100MB, you must use Git LFS (Large File Storage) to upload it.

Go to Streamlit Cloud:

Log in to share.streamlit.io with your GitHub account.

Click "New app" and select your repository, branch, and app.py as the main file.

Add Your API Key to Secrets:

In your app's settings, go to the "Secrets" section.

Paste your Gemini API key in the following format:

GEMINI_API_KEY = "your-actual-gemini-api-key-goes-here"


Save the secret. Your app will reboot and will now have secure access to the API
