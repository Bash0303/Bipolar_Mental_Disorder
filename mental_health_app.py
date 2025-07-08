import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
from datetime import datetime
import os

# Configuration
st.set_page_config(page_title=" Bipolar Mental Disorder Prediction", layout="wide", page_icon="🧠")

# Constants
DATA_FILE = 'mental_disorder_dataset.csv'
MODEL_FILE = 'mental_health_model.pkl'
HISTORY_FILE = 'prediction_history.csv'

# Expected columns (should match your dataset)
EXPECTED_COLUMNS = [
    "Patient Number", "Sadness", "Euphoric", "Exhausted", "Sleep dissorder",
    "Mood Swing", "Suicidal thoughts", "Anorxia", "Authority Respect",
    "Try-Explanation", "Aggressive Response", "Ignore & Move-On",
    "Nervous Break-down", "Admit Mistakes", "Overthinking", "Sexual Activity",
    "Concentration", "Optimisim", "Expert Diagnose"
]

# Define feature columns at the global level
FEATURE_COLS = [col for col in EXPECTED_COLUMNS if col not in ["Patient Number", "Expert Diagnose"]]

# Load or train model function
@st.cache_resource
def load_or_train_model():
    try:
        # Try to load existing model
        model, label_encoder, ordinal_encoder = pickle.load(open(MODEL_FILE, 'rb'))
        st.success("Loaded pre-trained model from disk")
        return model, label_encoder, ordinal_encoder
    except:
        st.warning("Training new model...")
        
        try:
            df = pd.read_csv(DATA_FILE)
            df.columns = df.columns.str.strip()
            
            # Check for missing columns
            missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
            if missing_cols:
                st.error(f"Missing columns in dataset: {missing_cols}")
                return None, None, None
            
            # Data preprocessing
            non_numeric_cols = df.select_dtypes(exclude=['number']).columns
            non_numeric_cols = [col for col in non_numeric_cols if col != "Expert Diagnose"]
            
            # Encode non-numeric features
            ordinal_encoder = OrdinalEncoder()
            if len(non_numeric_cols) > 0:
                df[non_numeric_cols] = ordinal_encoder.fit_transform(df[non_numeric_cols])
            
            # Encode target labels
            label_encoder = LabelEncoder()
            df["Diagnosis_Code"] = label_encoder.fit_transform(df["Expert Diagnose"])
            
            # Prepare features and target
            X = df.drop(columns=["Patient Number", "Expert Diagnose", "Diagnosis_Code"])
            y = df["Diagnosis_Code"]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.4, random_state=42, stratify=y
            )
            
            # Train model
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            
            # Save model
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump((model, label_encoder, ordinal_encoder), f)
            
            st.success("Model trained and saved successfully!")
            return model, label_encoder, ordinal_encoder
            
        except Exception as e:
            st.error(f"Error training model: {e}")
            return None, None, None

# Load prediction history
def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame()

# Save prediction to history
def save_to_history(input_data, prediction, probabilities):
    history_df = load_history()
    
    # Prepare new entry
    new_entry = input_data.copy()
    new_entry['Prediction'] = prediction
    new_entry['Prediction_DateTime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add probability columns
    for i, prob in enumerate(probabilities[0]):
        new_entry[f'Probability_Class_{i}'] = prob
    
    # Append to history
    if history_df.empty:
        history_df = pd.DataFrame([new_entry])
    else:
        history_df = pd.concat([history_df, pd.DataFrame([new_entry])], ignore_index=True)
    
    # Save to file
    history_df.to_csv(HISTORY_FILE, index=False)
    return history_df

# Clear history
def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return pd.DataFrame()

# Main app function
def main():
    st.title("🧠 Bipolar Mental Disorder Prediction System")
    st.markdown("""
    This application predicts mental health disorders based on behavioral and psychological indicators.
    Fill in the form below and click **Predict** to get a diagnosis prediction.
    """)
    
    # Load or train model
    model, label_encoder, ordinal_encoder = load_or_train_model()
    
    if model is None:
        st.error("Failed to load or train model. Please check your dataset.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Prediction", "History", "Model Info"])
    
    with tab1:
        st.header("Make a Prediction")
        
        # Create input form
        with st.form("prediction_form"):
            cols = st.columns(3)
            
            # Group inputs by category
            emotional_state = {
                "Sadness": cols[0].slider("Sadness", 0, 10, 5),
                "Euphoric": cols[1].slider("Euphoric", 0, 10, 5),
                "Exhausted": cols[2].slider("Exhausted", 0, 10, 5),
                "Mood Swing": cols[0].slider("Mood Swing", 0, 10, 5),
                "Optimisim": cols[1].slider("Optimisim", 0, 10, 5)
            }
            
            sleep_thoughts = {
                "Sleep dissorder": cols[2].slider("Sleep disorder", 0, 10, 5),
                "Suicidal thoughts": cols[0].slider("Suicidal thoughts", 0, 10, 5),
                "Overthinking": cols[1].slider("Overthinking", 0, 10, 5),
                "Concentration": cols[2].slider("Concentration", 0, 10, 5)
            }
            
            behavior = {
                "Anorxia": cols[0].slider("Anorxia", 0, 10, 5),
                "Authority Respect": cols[1].slider("Authority Respect", 0, 10, 5),
                "Try-Explanation": cols[2].slider("Try-Explanation", 0, 10, 5),
                "Aggressive Response": cols[0].slider("Aggressive Response", 0, 10, 5),
                "Ignore & Move-On": cols[1].slider("Ignore & Move-On", 0, 10, 5),
                "Nervous Break-down": cols[2].slider("Nervous Break-down", 0, 10, 5),
                "Admit Mistakes": cols[0].slider("Admit Mistakes", 0, 10, 5),
                "Sexual Activity": cols[1].slider("Sexual Activity", 0, 10, 5)
            }
            
            # Combine all inputs
            input_data = {**emotional_state, **sleep_thoughts, **behavior}
            
            # Add patient number (random for demo)
            input_data["Patient Number"] = np.random.randint(10000, 99999)
            
            submitted = st.form_submit_button("Predict")
        
        if submitted:
            # Prepare input for prediction
            input_df = pd.DataFrame([input_data])
            
            # Reorder columns to match training data
            input_df = input_df[["Patient Number"] + FEATURE_COLS]
            
            # Make prediction
            try:
                features = input_df[FEATURE_COLS]
                
                # Encode non-numeric features if needed
                non_numeric_cols = features.select_dtypes(exclude=['number']).columns
                if len(non_numeric_cols) > 0 and ordinal_encoder is not None:
                    features[non_numeric_cols] = ordinal_encoder.transform(features[non_numeric_cols])
                
                # Predict
                prediction_code = model.predict(features)[0]
                prediction = label_encoder.inverse_transform([prediction_code])[0]
                probabilities = model.predict_proba(features)
                
                # Save to history
                save_to_history(input_data, prediction, probabilities)
                
                # Display results
                st.success("Prediction completed successfully!")
                
                # Show prediction
                st.subheader("Prediction Result")
                cols = st.columns(2)
                cols[0].metric("Predicted Diagnosis", prediction)
                
                # Show probabilities
                st.subheader("Class Probabilities")
                prob_df = pd.DataFrame({
                    "Diagnosis Class": label_encoder.classes_,
                    "Probability": probabilities[0]
                }).sort_values("Probability", ascending=False)
                
                # Display as bar chart and table
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.barplot(data=prob_df, x="Diagnosis Class", y="Probability", ax=ax)
                ax.set_title("Prediction Probabilities")
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)
                
                st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}))
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
    
    with tab2:
        st.header("Prediction History")
        
        history_df = load_history()
        
        if not history_df.empty:
            # Show history table
            st.dataframe(history_df)
            
            # Download button
            st.download_button(
                label="Download History as CSV",
                data=history_df.to_csv(index=False).encode('utf-8'),
                file_name='mental_health_predictions.csv',
                mime='text/csv'
            )
            
            # Clear history button
            if st.button("Clear History"):
                history_df = clear_history()
                st.success("History cleared!")
                st.experimental_rerun()
        else:
            st.info("No prediction history found.")
    
    with tab3:
        st.header("Model Information")
        
        if model is not None:
            # Model details
            st.subheader("Model Type")
            st.write(f"Algorithm: {model.__class__.__name__}")
            st.write(f"Number of classes: {len(label_encoder.classes_)}")
            
            # Feature importance
            st.subheader("Feature Importance")
            if hasattr(model, 'coef_'):
                importance_df = pd.DataFrame({
                    "Feature": FEATURE_COLS,
                    "Importance": model.coef_[0]
                }).sort_values("Importance", ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax)
                ax.set_title("Feature Importance (Logistic Regression Coefficients)")
                st.pyplot(fig)
                
                st.dataframe(importance_df)
            else:
                st.warning("Feature importance not available for this model type.")
            
            # Class labels
            st.subheader("Class Labels")
            st.write(label_encoder.classes_)
        else:
            st.warning("No model information available.")

if __name__ == "__main__":
    main()