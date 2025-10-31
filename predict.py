import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
import warnings
warnings.filterwarnings('ignore')

class FakeNewsPredictor:
    def __init__(self, model_path='fake_news_model.pkl'):
        """Initialize the predictor with saved model"""
        try:
            # Download NLTK data if not already downloaded
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
            
            # Load the saved model components
            with open(model_path, 'rb') as file:
                model_data = pickle.load(file)
            
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.model_name = model_data['model_name']
            
            print(f"Loaded {self.model_name} model successfully!")
            
        except FileNotFoundError:
            print(f"Error: Model file '{model_path}' not found!")
            print("Please run 'train_model.py' first to create the model.")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]
        
        # Stemming
        stemmer = PorterStemmer()
        stemmed_words = [stemmer.stem(word) for word in filtered_words]
        
        return ' '.join(stemmed_words)
    
    def predict_single_news(self, text):
        """Predict if a single news article is fake or real"""
        try:
            # Preprocess the text
            processed_text = self.preprocess_text(text)
            
            if not processed_text.strip():
                return {
                    'error': 'Empty or invalid text provided',
                    'prediction': None,
                    'confidence': None
                }
            
            # Convert to TF-IDF features
            text_tfidf = self.vectorizer.transform([processed_text])
            
            # Make prediction
            prediction = self.model.predict(text_tfidf)[0]
            probabilities = self.model.predict_proba(text_tfidf)[0]
            
            # Prepare result
            result = {
                'original_text': text,
                'prediction': 'Fake' if prediction == 1 else 'Real',
                'confidence': max(probabilities),
                'probabilities': {
                    'Real': probabilities[0],
                    'Fake': probabilities[1]
                },
                'model_used': self.model_name
            }
            
            return result
            
        except Exception as e:
            return {
                'error': f'Prediction error: {e}',
                'prediction': None,
                'confidence': None
            }
    
    def predict_batch_news(self, news_list):
        """Predict multiple news articles at once"""
        results = []
        
        for i, news_text in enumerate(news_list):
            print(f"Processing article {i+1}/{len(news_list)}...")
            result = self.predict_single_news(news_text)
            results.append(result)
        
        return results
    
    def predict_from_file(self, file_path, text_column='text'):
        """Predict news from a CSV file"""
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            if text_column not in df.columns:
                return f"Error: Column '{text_column}' not found in the file."
            
            # Get predictions for all texts
            predictions = []
            confidences = []
            fake_probs = []
            
            print(f"Processing {len(df)} articles from file...")
            
            for idx, text in enumerate(df[text_column]):
                if idx % 50 == 0:  # Progress indicator
                    print(f"Processed {idx}/{len(df)} articles...")
                
                result = self.predict_single_news(text)
                
                if result.get('error'):
                    predictions.append('Error')
                    confidences.append(0)
                    fake_probs.append(0)
                else:
                    predictions.append(result['prediction'])
                    confidences.append(result['confidence'])
                    fake_probs.append(result['probabilities']['Fake'])
            
            # Add results to dataframe
            df['predicted_label'] = predictions
            df['confidence'] = confidences
            df['fake_probability'] = fake_probs
            
            # Save results
            output_file = file_path.replace('.csv', '_predictions.csv')
            df.to_csv(output_file, index=False)
            
            print(f"Predictions saved to: {output_file}")
            return df
            
        except Exception as e:
            return f"Error processing file: {e}"

def interactive_prediction():
    """Interactive mode for testing individual news articles"""
    try:
        predictor = FakeNewsPredictor()
        
        print("\n" + "="*60)
        print("FAKE NEWS DETECTION - INTERACTIVE MODE")
        print("="*60)
        print("Enter news headlines or articles to check if they're fake or real.")
        print("Type 'quit' to exit, 'batch' for batch testing, or 'file' for file processing.")
        print("="*60)
        
        while True:
            print("\nOptions:")
            print("1. Enter single news text")
            print("2. Batch test with sample news")
            print("3. Process CSV file")
            print("4. Quit")
            
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                news_text = input("\nEnter news text: ").strip()
                
                if news_text.lower() == 'quit':
                    break
                
                if not news_text:
                    print("Please enter some text.")
                    continue
                
                print("\nAnalyzing...")
                result = predictor.predict_single_news(news_text)
                
                if result.get('error'):
                    print(f"Error: {result['error']}")
                else:
                    print(f"\nPrediction: {result['prediction']}")
                    print(f"Confidence: {result['confidence']:.2%}")
                    print(f"Probabilities:")
                    print(f"  Real News: {result['probabilities']['Real']:.2%}")
                    print(f"  Fake News: {result['probabilities']['Fake']:.2%}")
                    print(f"Model Used: {result['model_used']}")
            
            elif choice == '2':
                sample_news = [
                    "Scientists discover cure for all types of cancer using common household item",
                    "Stock market reaches new record high amid positive economic indicators",
                    "Local man claims he can communicate with dolphins using special device",
                    "University researchers develop new renewable energy storage technology",
                    "Celebrity endorses dangerous weight loss pill that causes heart problems"
                ]
                
                print(f"\nTesting {len(sample_news)} sample articles...")
                results = predictor.predict_batch_news(sample_news)
                
                print("\n" + "="*80)
                print("BATCH PREDICTION RESULTS")
                print("="*80)
                
                for i, (news, result) in enumerate(zip(sample_news, results), 1):
                    print(f"\nArticle {i}:")
                    print(f"Text: {news}")
                    print(f"Prediction: {result['prediction']}")
                    print(f"Confidence: {result['confidence']:.2%}")
                    print("-" * 40)
            
            elif choice == '3':
                file_path = input("Enter CSV file path: ").strip()
                text_column = input("Enter text column name (default: 'text'): ").strip()
                
                if not text_column:
                    text_column = 'text'
                
                print("Processing file...")
                result = predictor.predict_from_file(file_path, text_column)
                
                if isinstance(result, str):  # Error message
                    print(result)
                else:
                    print("File processed successfully!")
                    print(f"Predictions summary:")
                    print(result['predicted_label'].value_counts())
            
            elif choice == '4':
                break
            
            else:
                print("Invalid option. Please select 1-4.")
        
        print("\nThank you for using Fake News Detector!")
        
    except Exception as e:
        print(f"Error initializing predictor: {e}")

def command_line_prediction(text):
    """Command line interface for single prediction"""
    try:
        predictor = FakeNewsPredictor()
        result = predictor.predict_single_news(text)
        
        if result.get('error'):
            print(f"Error: {result['error']}")
        else:
            print(f"Text: {text}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Real News Probability: {result['probabilities']['Real']:.2%}")
            print(f"Fake News Probability: {result['probabilities']['Fake']:.2%}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode with text argument
        news_text = ' '.join(sys.argv[1:])
        command_line_prediction(news_text)
    else:
        # Interactive mode
        interactive_prediction()
