import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Descriptors3D
from rdkit import RDLogger
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pickle
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor

RDLogger.DisableLog('rdApp.*')
st.title("Retention Time Prediction Tool")

with st.expander("❗ Instructions"):
    st.markdown("""
    This tool uses machine learning to help predict LC retention time (RT) by creating a model that describes the relationship between chemical structure and measured RT for a given LC method.

    **Overview**
    - You upload a spreadsheet with molecular structures in the form of SMILES strings and their corresponding experimental RTs.
    - The tool automatically calculates chemical features (called _descriptors_) for each molecule, such as molecular weight, LogP, no. of H-bond donors, etc.
    - Multiple, suitable machine learning models are trained on these descriptors and the corresponding experimental RTs to describe a mathematical relationship between the two.
    - These models are tested to find the one that gives the best predictions of RT from chemical structure.
    - Once the best model is found, you can download it and use it later to predict RTs for new molecules.

    **Important Notes**
    - 📊 **Data Size Matters**: Machine learning models can only learn useful patterns if there's enough high-quality data. A small or noisy dataset may not produce reliable predictions. The minimum input should be 10 chemical structures.
    - ⚗️ **Method-Specific**: The model is specific to the LC method used to generate the experimental RTs you provide. It can’t generalize to different chromatography setups.
    - 🧪 **Chemical Space**: The model will only work well for molecules that are chemically similar to those it was trained on. Predictions for very different molecules may be inaccurate.
                
    """)

with st.expander("🔎 How it works"):
    st.markdown("""              
    _Descriptor Calculation:_
    - Uses RDKit to compute 2D and 3D molecular descriptors.
    - 3D descriptors rely on conformer generation (ETKDGv3) and MMFF optimization.
    - Molecules that fail 3D embedding or optimization are excluded from training.

    _Preprocessing Optimization:_
    - Descriptors with variance below or correlation above a given threshold are removed to reduce noise and multicollinearity.
    - Low variance thresholds 0.01 to 0.15 tested.
    - High correlation thresholds 0.80 to 0.99 tested.

    _Feature Scaling:_
    - All descriptor data is scaled using StandardScaler to normalize feature distributions before training.

    _Model Training and Selection:_
    - Trains and evaluates Ridge, Lasso, and ElasticNet regressors.
    - Hyperparameters are tuned using RandomizedSearchCV with 5-fold cross-validation.
    - Scoring is based on negative mean squared error (MSE).
    - The model with the lowest cross-validated MSE is selected as the best.

    _Training/Test Split for Evaluation:_
    - A standard train_test_split (40% test set) is applied after model selection to assess overfitting visually.

    _Model Export:_
    - The final model is bundled with its associated feature list and scaler. This ensures consistent preprocessing and prediction for new data.

    _Prediction Mode:_
    - Input SMILES are transformed using the same descriptors and filtered/scaled using the saved configuration.
    - Only the features used during training are applied during inference.
    """)

# ------------- Descriptor Calculation -------------
def compute_3d_descriptors(mol):
    params = AllChem.ETKDGv3()
    success = AllChem.EmbedMolecule(mol, params)
    if success != 0:
        return None
    AllChem.MMFFOptimizeMolecule(mol)
    return Descriptors3D.CalcMolDescriptors3D(mol)

def compute_2d_descriptors(mol):
    return Descriptors.CalcMolDescriptors(mol)

def process_single_smiles(i, smiles, rt):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        mol = Chem.AddHs(mol)
        try:
            Chem.SanitizeMol(mol)
            descriptors_3d = compute_3d_descriptors(mol)
            if descriptors_3d is None:
                return None
            descriptors_2d = compute_2d_descriptors(mol)
            descriptors = {**descriptors_2d, **descriptors_3d}
            return (i, descriptors)
        except Exception:
            return None
    return None

def compute_training_descriptors(smiles_list, rt_list):
    data = []
    valid_idx = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_smiles, i, s, rt_list[i]) for i, s in enumerate(smiles_list)]
        for future in futures:
            result = future.result()
            if result:
                i, descriptors = result
                data.append(descriptors)
                valid_idx.append(i)
    df = pd.DataFrame(data)
    df.replace("", float("NaN"), inplace=True)
    df.dropna(how='all', axis=1, inplace=True)
    return df, valid_idx

def compute_prediction_descriptors(smiles_list):
    data = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol = Chem.AddHs(mol)
            try:
                Chem.SanitizeMol(mol)
                descriptors_3d = compute_3d_descriptors(mol)
                if descriptors_3d is None:
                    continue
                descriptors_2d = compute_2d_descriptors(mol)
                descriptors = {**descriptors_2d, **descriptors_3d}
                data.append(descriptors)
            except Exception:
                continue
    df = pd.DataFrame(data)
    df.replace("", float("NaN"), inplace=True)
    df.dropna(how='all', axis=1, inplace=True)
    return df

# ----------------------------- Data Cleaning -----------------------------
def clean_descriptor_data(df, var_thresh, corr_thresh):
    var_vals = df.var(numeric_only=True)
    low_var = var_vals[var_vals < var_thresh].index
    df.drop(columns=low_var, inplace=True)

    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > corr_thresh)]
    df.drop(columns=to_drop, inplace=True)

    return df

# ------------- Model Training -------------
def train_models(X, y):
    models = {
        "Ridge": Ridge(),
        "Lasso": Lasso(max_iter=10000),
        "ElasticNet": ElasticNet(max_iter=10000)
    }

    params = {
        "Ridge": {"alpha": np.logspace(-4, 4, 100)},
        "Lasso": {"alpha": np.logspace(-6, 2, 100)},
        "ElasticNet": {
            "alpha": np.logspace(-6, 2, 100),
            "l1_ratio": np.linspace(0.1, 1.0, 10)
        }
    }

    best_models = {}
    for name in models:
        search = RandomizedSearchCV(models[name], params[name], n_iter=25, scoring='neg_mean_squared_error', cv=5, random_state=42, n_jobs=-1)
        search.fit(X, y)
        best_models[name] = (search.best_estimator_, -search.best_score_, search.best_params_)

    return best_models

# ------------- Interface: Train and Predict -------------
mode = st.radio("", ["Train and Tune a Model", "Predict RT with a Saved Model"])

if mode == "Train and Tune a Model":
    file = st.file_uploader("Upload a CSV with **SMILES** and **RT** columns", type="csv")
    if file:
        data = pd.read_csv(file)
        if not {'SMILES', 'RT'}.issubset(data.columns):
            st.error("CSV must contain 'SMILES' and 'RT' columns.")
            st.stop()

        st.info("Computing descriptors...")
        df_desc, valid_idx = compute_training_descriptors(data['SMILES'], data['RT'])
        if df_desc.empty:
            st.error("Failed to generate descriptors for valid molecules.")
            st.stop()

        y = np.array(data['RT'])[valid_idx]

        st.info("Optimizing preprocessing thresholds...")
        best_score = float("inf")
        best_X = None
        best_params = (None, None)
        best_scaler = None
        progress = st.progress(0, text="Optimizing thresholds...")
        steps = 0
        total_steps = 10 * 10
        for var_thresh in np.linspace(0.01, 0.15, 10):
            for corr_thresh in np.linspace(0.8, 0.99, 10):
                temp_df = df_desc.copy()
                temp_df = clean_descriptor_data(temp_df, var_thresh, corr_thresh)
                if temp_df.shape[1] < 5:
                    continue
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(temp_df)
                models = train_models(X_scaled, y)
                min_model = min(models.items(), key=lambda x: x[1][1])
                if min_model[1][1] < best_score:
                    best_score = min_model[1][1]
                    best_X = X_scaled
                    best_scaler = scaler
                    best_model_name = min_model[0]
                    best_model = min_model[1][0]
                    best_params = min_model[1][2]
                    best_features = temp_df.columns
                steps += 1
                progress.progress(min(steps / total_steps, 1.0))
        progress.empty()

        st.success(f"Best Model: {best_model_name} (MSE: {best_score:.3f})")

        X = df_desc[best_features].values
        X_train_raw, X_test_raw, y_train_plot, y_test_plot = train_test_split(X, y, test_size=0.4, random_state=42)
        X_train_plot = best_scaler.transform(X_train_raw)
        X_test_plot = best_scaler.transform(X_test_raw)

        y_train_pred = best_model.predict(X_train_plot)
        y_test_pred = best_model.predict(X_test_plot)

        fig, ax = plt.subplots()
        ax.scatter(y_train_plot, y_train_pred, alpha=0.6, label="Train", color="blue")
        ax.scatter(y_test_plot, y_test_pred, alpha=0.6, label="Test", color="orange")
        ax.plot([min(y), max(y)], [min(y), max(y)], 'r--')
        ax.set_xlabel("Experimental RT")
        ax.set_ylabel("Predicted RT")
        ax.set_title(f"{best_model_name} Prediction Performance")
        ax.legend()
        st.pyplot(fig)

        model_bundle = {
            'model': best_model,
            'features': list(best_features),
            'scaler': best_scaler
        }

        st.download_button(
            "Download Best Model",
            data=pickle.dumps(model_bundle),
            file_name=f"{best_model_name}_model.pkl",
            mime='application/octet-stream'
        )

elif mode == "Predict RT with a Saved Model":
    model_file = st.file_uploader("Upload saved model (.pkl)", type="pkl")
    smiles_file = st.file_uploader("Upload SMILES CSV", type="csv")

    if model_file and smiles_file:
        model_bundle = pickle.load(model_file)
        smiles_df = pd.read_csv(smiles_file)

        if 'SMILES' not in smiles_df.columns:
            st.error("CSV must contain a 'SMILES' column.")
            st.stop()

        df_desc = compute_prediction_descriptors(smiles_df['SMILES'])
        if df_desc.empty:
            st.error("No valid descriptors generated.")
            st.stop()

        df_desc = df_desc[model_bundle['features']]
        df_scaled = model_bundle['scaler'].transform(df_desc)
        preds = model_bundle['model'].predict(df_scaled)

        result_df = pd.DataFrame({
            "SMILES": smiles_df['SMILES'].iloc[:len(preds)].values,
            "Predicted RT": preds
        })
        st.dataframe(result_df)

        st.download_button(
            "Download Predictions",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="predicted_rt.csv",
            mime="text/csv"
        )
