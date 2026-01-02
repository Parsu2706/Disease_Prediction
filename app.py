import streamlit as st
import plotly.express as px
import pickle
import numpy as np
import pandas as pd
import json
import torch

from model import DiseasePrediction

@st.cache_resource
def load_model_meta():
    with open("assets/model_meta.json", "r") as f:
        return json.load(f)

@st.cache_resource
def load_model():
    meta = load_model_meta()
    model = DiseasePrediction(input_size=meta['input_size'] , output_size=meta['output_size'])
    model.load_state_dict(torch.load("assets/disease_final_model.pt" , map_location='cpu'))
    model.eval()
    return model


@st.cache_resource
def load_label_encoder():
    path = "assets/label_encoder.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_symptoms(): 
    with open("assets/symptoms.json" , 'r') as f : 
        return json.load(f)

@st.cache_resource
def load_feature_names(): 
    with open("assets/feature_names.json" , "r") as f : 
        return json.load(f)
    

def predict_and_top_k(k, model, X_input, encoder):
    with torch.no_grad(): 
        logits = model(X_input)
        probs = torch.softmax(logits , dim=1).cpu().numpy()
    top_index = probs.argsort(axis=1)[:, -k:][:, ::-1]
    top_index = top_index.squeeze()
    top_probs = probs[0 , top_index]
    top_labels = encoder.inverse_transform(top_index)
    return top_labels, top_probs

if __name__ == "__main__":
    st.set_page_config(page_title="Disease Prediction" , layout='centered')
 
    model = load_model()
    le = load_label_encoder()
    symptoms = load_symptoms()
    feature_names = load_feature_names()
    feature_index = {name : i for i , name in  enumerate(feature_names)}

    st.title("Disease Prediction Dashboard")

    st.sidebar.header("Patient Input")

    st.sidebar.file_uploader("Upload CSV")

    top_k = st.sidebar.selectbox("Show Top K Prediction" , [1 , 3 , 5 ,7 ,9 ,10] , index=2)
    confidence_threshold = st.sidebar.slider("Confidence_threshold" , min_value=0.0 , max_value=1.0 , value=0.5 , step=0.05)

    tab1 , tab2 = st.tabs(['Select Symptomns', "Predictions"])
    with tab1:
        col1, col2 = st.columns([1, 4])
        with col1:
            clear_clicked = st.button("Clear all")
        if clear_clicked:
            for main_symp in symptoms.keys():
                st.session_state[main_symp] = []
            st.rerun()

        select_symptoms = st.sidebar.button("Select Symptoms")
        selected_symptoms = []
        for main_symp , symp in symptoms.items() : 
            with st.expander(main_symp): 
                chosen = st.multiselect("select symptoms" , symp , key=main_symp)
                selected_symptoms.extend(chosen)
        st.info(f"Selected symptoms: {len(selected_symptoms)}")

    with tab2: 
        if st.sidebar.button("Predict"): 
            if len(selected_symptoms) == 0 : 
                st.error("Please select the symptoms")
                st.stop()
        
        x_np = np.zeros((1 , len(feature_names)) , dtype=np.float32)
        
        for symptom in selected_symptoms: 
            if symptom in feature_index:
                x_np[0, feature_index[symptom]] = 1.0


        x_inputs = torch.from_numpy(x_np).float()

        labels , probs = predict_and_top_k(top_k , model , x_inputs , le)


        st.subheader(f"Top-{top_k} Predicted Diseases")
        result_df = pd.DataFrame({"Disease" : labels , "Proability" : probs})
        
        if probs[0] < confidence_threshold : 
            st.warning("Consider further test or expert review")

        fig = px.bar(x = labels , y = probs , labels={'x' : 'Disease' , 'y': "Probability"} , title="Prediction Confidence")
        st.plotly_chart(fig , use_container_width=True)
