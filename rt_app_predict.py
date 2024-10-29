import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Descriptors3D
import pickle

st.title("HPLC Retention Time Prediction Using Pretrained Model")

def compute_3d_descriptors(mol):
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    return Descriptors3D.CalcMolDescriptors3D(mol)

def compute_2d_descriptors(mol):
    return Descriptors.CalcMolDescriptors(mol)

def descriptors_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        mol = Chem.AddHs(mol)
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            st.error(f"Sanitization failed: {e}")
            return None

        descriptors_3d = compute_3d_descriptors(mol)
        descriptors_2d = compute_2d_descriptors(mol)

        descriptors = {**descriptors_2d, **descriptors_3d}
        df = pd.DataFrame([descriptors])
        df.replace("", float("NaN"), inplace=True)
        df.dropna(how='all', axis=1, inplace=True)
        
        return df
    else:
        st.error("Invalid SMILES string.")
        return None

uploaded_model = st.file_uploader("Upload a trained model (pickle file)", type="pkl")
smiles_input = st.text_input("Enter SMILES string")

if uploaded_model is not None and smiles_input:
    # Load the model and extract expected features
    model = pickle.load(uploaded_model)
    
    if hasattr(model, 'coef_'):
        expected_features = model.feature_names_in_  # Works for linear models in scikit-learn
    elif hasattr(model, 'feature_importances_'):
        expected_features = model.feature_names_in_  # Works for tree-based models in scikit-learn
    else:
        # If feature names were saved separately, they could be stored in the pickle or loaded alongside the model
        expected_features = list(model.get_booster().feature_names)
        
    # Generate descriptors from the input SMILES
    descriptors_df = descriptors_from_smiles(smiles_input)
    
    if descriptors_df is not None:
        # Filter the descriptors to match the expected features
        descriptors_df_filtered = descriptors_df.reindex(columns=expected_features, fill_value=0)
        
        # Predict using the model
        prediction = model.predict(descriptors_df_filtered)
        st.write(f"Predicted Retention Time (RT): {prediction[0]:.2f} minutes")
    else:
        st.error("Failed to generate descriptors from the SMILES string.")
else:
    st.info("Please upload a trained model and enter a valid SMILES string.")
