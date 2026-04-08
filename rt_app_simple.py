import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Descriptors3D
from rdkit import RDLogger
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

RDLogger.DisableLog('rdApp.*')

# ----------------------------- UI -----------------------------
st.image('./catsci-logo.svg', width=200)
st.title("Retention Time Predictor")
st.write("""Train and use simple regression models for predicting reverse phase LC retention times. 
Works best with datasets containing >10 similar compounds with corresponding RTs acquired on a single LC method.""")

# ----------------------------- Descriptor Calculation -----------------------------
def compute_3d_descriptors(mol):
    params = AllChem.ETKDGv3()
    success = AllChem.EmbedMolecule(mol, params)
    if success != 0:
        return None
    AllChem.MMFFOptimizeMolecule(mol)
    return Descriptors3D.CalcMolDescriptors3D(mol)

def compute_2d_descriptors(mol):
    return Descriptors.CalcMolDescriptors(mol)

def process_single_smiles(i, smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        mol = Chem.AddHs(mol)
        try:
            Chem.SanitizeMol(mol)
            d3 = compute_3d_descriptors(mol)
            if d3 is None:
                return None
            d2 = compute_2d_descriptors(mol)
            return (i, {**d2, **d3})
        except Exception:
            return None
    return None

def compute_training_descriptors(smiles_list):
    data, valid_idx = [], []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_smiles, i, s) for i, s in enumerate(smiles_list)]
        for future in futures:
            result = future.result()
            if result:
                i, descriptors = result
                data.append(descriptors)
                valid_idx.append(i)
    df = pd.DataFrame(data)
    df.replace("", np.nan, inplace=True)
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
                d3 = compute_3d_descriptors(mol)
                if d3 is None:
                    continue
                d2 = compute_2d_descriptors(mol)
                data.append({**d2, **d3})
            except Exception:
                continue
    df = pd.DataFrame(data)
    df.replace("", np.nan, inplace=True)
    df.dropna(how='all', axis=1, inplace=True)
    return df

# ----------------------------- Custom Transformer -----------------------------
class DescriptorCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, var_thresh=0.05, corr_thresh=0.95):
        self.var_thresh = var_thresh
        self.corr_thresh = corr_thresh

    def fit(self, X, y=None):
        X = pd.DataFrame(X)

        var_vals = X.var(numeric_only=True)
        self.low_var_cols_ = var_vals[var_vals < self.var_thresh].index.tolist()

        X_reduced = X.drop(columns=self.low_var_cols_, errors='ignore')

        corr_matrix = X_reduced.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        self.corr_cols_ = [
            col for col in upper_triangle.columns
            if any(upper_triangle[col] > self.corr_thresh)
        ]

        self.final_columns_ = X_reduced.drop(columns=self.corr_cols_, errors='ignore').columns.tolist()
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X = X.drop(columns=self.low_var_cols_, errors='ignore')
        X = X.drop(columns=self.corr_cols_, errors='ignore')

        # Align columns (important for prediction)
        X = X.reindex(columns=self.final_columns_, fill_value=0)

        return X.values

# ----------------------------- Model Training -----------------------------
def train_models(X_train, y_train):
    models = {
        "Ridge": Ridge(),
        "Lasso": Lasso(max_iter=50000),
        "ElasticNet": ElasticNet(max_iter=50000),
    }

    params = {
        "Ridge": {"model__alpha": np.logspace(-4, 4, 100)},
        "Lasso": {"model__alpha": np.logspace(-6, 2, 100)},
        "ElasticNet": {
            "model__alpha": np.logspace(-6, 2, 100),
            "model__l1_ratio": np.linspace(0.1, 1.0, 10)
        },
    }

    best_models = {}

    for name in models:
        pipe = Pipeline([
            ("clean", DescriptorCleaner()),
            ("scale", StandardScaler()),
            ("model", models[name])
        ])

        search = RandomizedSearchCV(
            pipe,
            params[name],
            n_iter=200,
            scoring='neg_root_mean_squared_error',
            cv=5,
            n_jobs=1,
            random_state=42
        )

        search.fit(X_train, y_train)

        best_models[name] = (
            search.best_estimator_,
            search.best_score_,
            search.best_params_
        )

    return best_models

# ----------------------------- Interface -----------------------------
mode = st.radio("", ["Train and Tune a Model", "Predict RT with a Saved Model"])

# ----------------------------- TRAIN -----------------------------
if mode == "Train and Tune a Model":
    file = st.file_uploader("Upload CSV with SMILES and RT", type="csv")

    if file:
        data = pd.read_csv(file)

        if not {'SMILES', 'RT'}.issubset(data.columns):
            st.error("CSV must contain SMILES and RT")
            st.stop()

        st.info("Computing descriptors...")
        df_desc, valid_idx = compute_training_descriptors(data['SMILES'])

        if df_desc.empty:
            st.error("Descriptor calculation failed")
            st.stop()

        y = np.array(data['RT'])[valid_idx]

        # SPLIT FIRST (no leakage)
        X_train, X_test, y_train, y_test = train_test_split(
            df_desc, y, test_size=0.35, random_state=42
        )

        st.info("Training models...")
        models = train_models(X_train, y_train)

        best_name, (best_model, best_score, best_params) = max(
            models.items(), key=lambda x: x[1][1]
        )

        st.success(f"Best Model: {best_name} (CV RMSE: {best_score:.4f})")

        # Test evaluation
        y_pred = best_model.predict(X_test)

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([min(y), max(y)], [min(y), max(y)], 'r--')
        ax.set_xlabel("Experimental RT")
        ax.set_ylabel("Predicted RT")
        ax.set_title(best_name)
        st.pyplot(fig)

        model_bundle = {'model': best_model}

        st.download_button(
            "Download Model",
            data=pickle.dumps(model_bundle),
            file_name=f"{best_name}_model.pkl"
        )

# ----------------------------- PREDICT -----------------------------
elif mode == "Predict RT with a Saved Model":
    model_file = st.file_uploader("Upload model", type="pkl")
    smiles_file = st.file_uploader("Upload SMILES CSV", type="csv")

    if model_file and smiles_file:
        model_bundle = pickle.load(model_file)
        smiles_df = pd.read_csv(smiles_file)

        if 'SMILES' not in smiles_df.columns:
            st.error("SMILES column missing")
            st.stop()

        df_desc = compute_prediction_descriptors(smiles_df['SMILES'])

        if df_desc.empty:
            st.error("Descriptor generation failed")
            st.stop()

        preds = model_bundle['model'].predict(df_desc)

        result_df = pd.DataFrame({
            "SMILES": smiles_df['SMILES'].iloc[:len(preds)].values,
            "Predicted RT": preds
        })

        st.dataframe(result_df)

        st.download_button(
            "Download Predictions",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv"
        )
