from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import asyncio

# Load the trained SVM model
svm_model = joblib.load('svm_model.pkl')

# Initialize FastAPI app
app = FastAPI()

# Simulating the TF-IDF vectorizer (or load if you saved it)
vectorizer = TfidfVectorizer()

# Example model input format
class Document(BaseModel):
    text: str

# Function to clean and limit text
def clean_text(text: str, max_length: int = 500):
    # Remove unwanted characters (e.g., special characters, numbers, etc.)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Limit the text length
    text = text[:max_length]  # Limit the text to max_length characters
    
    return text

# Define a prediction endpoint
@app.post("/predict/")
async def predict(document: Document):
    # Simulate an asynchronous delay (e.g., time for preprocessing or model inference)
    await asyncio.sleep(1)
    
    # Clean and limit the text
    cleaned_text = clean_text(document.text)
    
    # Transform the cleaned text using the same vectorizer used during training
    text_vector = vectorizer.transform([cleaned_text])
    
    # Predict using the SVM model
    prediction = await asyncio.to_thread(svm_model.predict, text_vector)
    
    # Return the prediction
    return {"prediction": str(prediction[0])}

