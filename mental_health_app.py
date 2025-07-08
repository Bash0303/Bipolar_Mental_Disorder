import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from datetime import datetime
import os

# Configuration
st.set_page_config(page_title="Mental Disorder Prediction", layout="wide", page_icon="🧠")

# Constants
DATA_FILE = 'mental_disorder_dataset.csv'
MODEL_FILE = 'mental_health_model.pkl'
HISTORY_FILE = 'prediction_history.csv'

# Expected columns
EXPECTED_COLUMNS = [
    "Patient Number", "Sadness", "Euphoric", "Exhausted", "Sleep dissorder",
    "Mood Swing", "Suicidal thoughts", "Anorxia", "Authority Respect",
    "Try-Explanation", "Aggressive Response", "Ignore & Move-On",
    "Nervous Break-down", "Admit Mistakes", "Overthinking", "Sexual Activity",
    "Concentration", "Optimisim", "Expert Diagnose"
]

# Feature columns (defined globally)
FEATURE_COLS = [col for col in EXPECTED_COLUMNS if col not in ["Patient Number", "Expert Diagnose"]]

# Load or train model
@st.cache_resource
def load_or_train_model():
    try:
        with open(MODEL_FILE, 'rb') as f:
            model, label_encoder, ordinal_encoder = pickle.load(f)
        st.success("Loaded pre-trained model")
        return model, label_encoder, ordinal_encoder
    except:
        st.warning("Training new model...")
        try:
            df = pd.read_csv(DATA_FILE)
            df.columns = df.columns.str.strip()
            
            missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
                return None, None, None
            
            non_numeric_cols = df.select_dtypes(exclude=['number']).columns
            non_numeric_cols = [col for col in non_numeric_cols if col != "Expert Diagnose"]
            
            ordinal_encoder = OrdinalEncoder()
            if non_numeric_cols:
                df[non_numeric_cols] = ordinal_encoder.fit_transform(df[non_numeric_cols])
            
            label_encoder = LabelEncoder()
            df["Diagnosis_Code"] = label_encoder.fit_transform(df["Expert Diagnose"])
            
            X = df[FEATURE_COLS]
            y = df["Diagnosis_Code"]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.4, random_state=42, stratify=y
            )
            
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump((model, label_encoder, ordinal_encoder), f)
            
            st.success("Model trained successfully!")
            return model, label_encoder, ordinal_encoder
        except Exception as e:
            st.error(f"Training error: {str(e)}")
            return None, None, None

# Load history
@st.cache_data
def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame()

# Save prediction
def save_to_history(input_data, prediction, probabilities):
    history_df = load_history()
    new_entry = input_data.copy()
    new_entry['Prediction'] = prediction
    new_entry['DateTime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for i, prob in enumerate(probabilities[0]):
        new_entry[f'Prob_Class_{i}'] = prob
    
    history_df = pd.concat([history_df, pd.DataFrame([new_entry])], ignore_index=True)
    history_df.to_csv(HISTORY_FILE, index=False)
    return history_df

# Main app
def main():
    st.title("🧠 Mental Disorder Prediction System")
    st.markdown("Predict mental health disorders based on behavioral indicators.")
    
    model, label_encoder, ordinal_encoder = load_or_train_model()
    if not model:
        return
        
    tab1, tab2, tab3 = st.tabs(["Predict", "History", "Model Info"])
    
    with tab1:
        st.header("Make Prediction")
        with st.form("prediction_form"):
            cols = st.columns(3)
            input_data = {
                "Sadness": cols[0].slider("Sadness", 0, 10, 5),
                "Euphoric": cols[1].slider("Euphoric", 0, 10, 5),
                # ... [keep all your existing sliders] ...
                "Sexual Activity": cols[1].slider("Sexual Activity", 0, 10, 5)
            }
            input_data["Patient Number"] = np.random.randint(10000, 99999)
            
            if st.form_submit_button("Predict"):
                input_df = pd.DataFrame([input_data])
                features = input_df[FEATURE_COLS]
                
                if ordinal_encoder:
                    non_numeric = features.select_dtypes(exclude=['number']).columns
                    if non_numeric.any():
                        features[non_numeric] = ordinal_encoder.transform(features[non_numeric])
                
                prediction_code = model.predict(features)[0]
                prediction = label_encoder.inverse_transform([prediction_code])[0]
                probabilities = model.predict_proba(features)
                
                save_to_history(input_data, prediction, probabilities)
                
                st.success("Prediction complete!")
                st.metric("Diagnosis", prediction)
                
                prob_df = pd.DataFrame({
                    "Class": label_encoder.classes_,
                    "Probability": probabilities[0]
                }).sort_values("Probability", ascending=False)
                
                fig, ax = plt.subplots()
                sns.barplot(data=prob_df, x="Class", y="Probability", ax=ax)
                st.pyplot(fig)
                st.dataframe(prob_df)
    
    with tab2:
        st.header("History")
        history_df = load_history()
        if not history_df.empty:
            st.dataframe(history_df)
            st.download_button(
                "Download History",
                history_df.to_csv(index=False),
                "mental_health_history.csv"
            )
            if st.button("Clear History"):
                if os.path.exists(HISTORY_FILE):
                    os.remove(HISTORY_FILE)
                st.rerun()
        else:
            st.info("No history yet")
    
    with tab3:
        st.header("Model Info")
        st.write(f"Algorithm: {model.__class__.__name__}")
        if hasattr(model, 'coef_'):
            importance = pd.DataFrame({
                "Feature": FEATURE_COLS,
                "Importance": model.coef_[0]
            }).sort_values("Importance", ascending=False)
            st.dataframe(importance)
        st.write("Classes:", label_encoder.classes_)

if __name__ == "__main__":
    main()