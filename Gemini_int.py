from google import genai
from dotenv import load_dotenv
import streamlit as st 
import os 

from dotenv import load_dotenv

load_dotenv()

API_KEY = None
if "GEMINI_API_KEY" in st.secrets: 
    API_KEY = st.secrets["GEMINI_API_KEY"]
else : 
    API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY: 
    raise ValueError("GEMINI_API_KEY Not found.")
client = genai.Client(api_key=API_KEY)


def gemini_analysis(symptoms , predicted_diseases ,probs, model_name : str): 
    symptoms_str = ", ".join(symptoms)
    predicted_diseases_str = ", ".join(
        [f"{d} ({round(p*100,1)}%)" for d, p in zip(predicted_diseases, probs)]
    )
    prompt = f"""
    Role: Act as a Senior Diagnostic Physician.

    Patient Symptoms: {symptoms_str}
    Model Predictions: {predicted_diseases_str}

    Task:

    1. 🎯 Most Likely Diagnosis (highlight clearly)
    2. 🩺 Clinical Correlation
    3. 🧬 Pathophysiology
    4. ⚠️ Red Flags
    5. 💉 Diagnostic Roadmap
    6. 💊 Immediate Precautions

    Guidelines:
    - Avoid vague categories
    - Be concise
    - Use bullet points

    Disclaimer:
    This is AI-generated and not a medical diagnosis.
    """
    
    response = client.models.generate_content(model=model_name , contents=prompt, config= {"temperature" : 0.4} )
    if hasattr(response , "text") and response.text : 
    
        return response.text
    else: 
        return str(response)
    


