import streamlit as st
import plotly.express as px
import pickle
import numpy as np
import pandas as pd
import json
import torch
import os 
from dotenv import load_dotenv
from model import DiseasePrediction
from Gemini_int import gemini_analysis


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


@st.cache_resource
def load_model_meta():
    """
    Load the model metadata and cached it to avoid repeated loading 
    """
    with open("assets/model_meta.json", "r") as f:
        return json.load(f)

@st.cache_resource
def load_model():
    """ 
    Load the trainined model for disease prediction , use the metadata from load_model_data() , then initialize the model.
    Load the trained weights from file ..
    """

    #load metadata
    meta = load_model_meta()
    #load the model 
    model = DiseasePrediction(input_size=meta['input_size'] , output_size=meta['output_size'])
    #load the trained weights of model 
    model.load_state_dict(torch.load("assets/disease_final_model.pt" , map_location='cpu'))
    #set model for evaluation .. 
    model.eval()

    return model


@st.cache_resource
def load_label_encoder():
    """Load the label encoder  and cached it ."""
    path = "assets/label_encoder.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_symptoms(): 
    """ 
    load the symptoms which are used by model as a input ,
    """
    with open("assets/symptoms.json" , 'r') as f : 
        return json.load(f)

@st.cache_resource
def load_feature_names(): 

    with open("assets/feature_names.json" , "r") as f : 
        return json.load(f)
    

def predict_and_top_k(k, model, X_input, encoder):
    """ Use the rained model to predict only top k diasess , convert the predicted numbers back to the features(use label encoder)."""
    with torch.no_grad(): 
        logits = model(X_input)
        probs = torch.softmax(logits , dim=1).cpu().numpy()
    top_index = probs.argsort(axis=1)[:, -k:][:, ::-1]
    top_index = top_index.squeeze()
    top_probs = probs[0 , top_index]
    top_labels = encoder.inverse_transform(top_index)
    return top_labels, top_probs

if __name__ == "__main__":

    # Main dashboard
    st.set_page_config(page_title="Disease Prediction" , layout='centered')
 
    model = load_model()
    le = load_label_encoder()
    symptoms = load_symptoms()
    feature_names = load_feature_names()
    feature_index = {name : i for i , name in  enumerate(feature_names)}

    st.title("🩺 Disease Prediction Dashboard")

    st.sidebar.header("Patient Input")

    st.sidebar.file_uploader("Upload CSV (optional)")

    top_k = st.sidebar.selectbox("Show Top K Prediction" , [1 , 3 , 5 ,7 ,9 ,10] , index=2)
    confidence_threshold = st.sidebar.slider("Confidence_threshold" , min_value=0.0 , max_value=1.0 , value=0.5 , step=0.05)

    
    tab1 , tab2 = st.tabs(['Select Symptomns', "Predictions"])
    with tab1:
        col1, col2 = st.columns([1, 4])
        with col1:
            clear_clicked = st.button("Clear all")
        if clear_clicked:
            st.session_state.selected_symptoms = []
            st.rerun()
       
        if "selected_symptoms" not in st.session_state: 
            st.session_state['selected_symptoms'] = []
        selected = []
        for main_symp , symp in symptoms.items(): 
            with st.expander(main_symp): 
                chosen = st.multiselect(
                    "select symptoms",
                    symp,
                    default=[s for s in st.session_state.selected_symptoms if s in symp],
                    key=main_symp
                )
                selected.extend(chosen)
        if selected:
            st.session_state.selected_symptoms = selected
        st.markdown("---")
        st.write("")
        st.write("")
        st.info(f"Selected Symptoms : {len(st.session_state.selected_symptoms)}")
        with st.container(border= True ):
            st.write(f"Symptoms : {" ,".join(st.session_state.selected_symptoms)} ")

    with tab2: 
        if st.button("Predict"): 
            if len(st.session_state.selected_symptoms) == 0 : 
                st.error("Please Select Symptoms")
                st.stop()

            x_np = np.zeros((1 , len(feature_names)) , dtype=np.float32)
            
            for symptom in st.session_state.selected_symptoms: 
                if symptom in feature_index:
                    x_np[0, feature_index[symptom]] = 1.0


            x_inputs = torch.from_numpy(x_np).float()

            labels , probs = predict_and_top_k(top_k , model , x_inputs , le)


            st.subheader(f"Top-{top_k} Predicted Diseases")
            result_df = pd.DataFrame({"Disease" : labels , "Probability" : probs})
            
            if probs[0] < confidence_threshold : 
                st.warning("Consider further test or expert review")

            fig = px.bar(x = labels , y = probs , labels={'x' : 'Disease' , 'y': "Probability"} , title="Prediction Confidence")
            st.plotly_chart(fig , use_container_width=True)

            if GEMINI_API_KEY: 
                    
                with st.spinner("AI is analyzing the clinical data..."): 
                    analysis = gemini_analysis(st.session_state.selected_symptoms , labels[:3] , probs[:3],model_name='models/gemini-2.0-flash')

                    st.markdown("---")
                    st.header("👨🏻‍⚕️ Clinial AI Insights")
                    with st.container(border=True):
                        st.markdown(analysis)

        

