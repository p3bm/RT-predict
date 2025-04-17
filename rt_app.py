import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Descriptors3D
from rdkit import RDLogger
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, make_scorer, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
from scipy.stats import uniform, randint

RDLogger.DisableLog('rdApp.*')
st.title("HPLC Retention Time Prediction")

# ----------------------------- Descriptor Computation -----------------------------
def compute_3d_descriptors(mol):
    params = AllChem.ETKDGv3()
    success = AllChem.EmbedMolecule(mol, params)
    if success != 0:
        return None
    AllChem.MMFFOptimizeMolecule(mol)
    return Descriptors3D.CalcMolDescriptors3D(mol)

def compute_2d_descriptors(mol):
    return Descriptors.CalcMolDescriptors(mol)

@st.cache_data(show_spinner=False)
def compute_training_descriptors(smiles_list, rt_list):
    data = []
    valid_rts = []
    for i, smiles in enumerate(smiles_list):
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
                valid_rts.append(rt_list[i])
            except Exception:
                continue
    df = pd.DataFrame(data)
    df.replace("", float("NaN"), inplace=True)
    df.dropna(how='all', axis=1, inplace=True)
    return df, valid_rts

@st.cache_data(show_spinner=False)
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
    st.write(f"No. of descriptors after low var removal: {df.shape[1]}")

    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > corr_thresh)]
    df.drop(columns=to_drop, inplace=True)
    st.write(f"No. of descriptors after high corr removal: {df.shape[1]}")

    return df

# ----------------------------- Model Training -----------------------------
@st.cache_resource(show_spinner=True)
def train_all_models_cached(data, _model_dict, test_set_size, k, scale):
    def train_all_models(data, model_dict, test_set_size, k, scale):
        X = data.drop(columns=['RT'])
        y = data['RT']

        if scale:
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        else:
            scaler = None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=42)

        results = {}
        summary = []
        fig, axs = plt.subplots(3, 3, figsize=(16, 12))
        axs = axs.flatten()

        for idx, (name, model) in enumerate(model_dict.items()):
            mse_scorer = make_scorer(mean_squared_error)
            cv_scores = cross_val_score(model, X_train, y_train, cv=k, scoring=mse_scorer)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)
            test_mse = mean_squared_error(y_test, y_test_pred)

            results[name] = {
                "model": model,
                "cv_scores": cv_scores,
                "test_mse": test_mse,
                "results_df": pd.DataFrame({
                    'Experimental RT': y_test,
                    'Predicted RT': y_test_pred
                }),
                "features": list(X.columns),
                "scaler": scaler
            }

            summary.append({
                "Model": name,
                "Mean CV MSE": np.mean(cv_scores),
                "Test MSE": test_mse
            })

            ax = axs[idx]
            ax.scatter(y_train, y_train_pred, label='Train', alpha=0.6)
            ax.scatter(y_test, y_test_pred, label='Test', alpha=0.6)
            ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
            ax.set_title(f"{name}\nCV MSE: {np.mean(cv_scores):.3f}, Test MSE: {test_mse:.3f}")
            ax.set_xlabel("Experimental RT (min)")
            ax.set_ylabel("Predicted RT (min)")
            ax.legend()

        for j in range(len(model_dict), 9):
            fig.delaxes(axs[j])

        plt.tight_layout(pad=4.0)
        return results, pd.DataFrame(summary), fig

    return train_all_models(data, _model_dict, test_set_size, k, scale)

# ----------------------------- Hyperparameter Tuning -----------------------------
@st.cache_resource(show_spinner=True)
def tune_model_cached(model_name, _param_distributions, X, y):
    def tune_model(model_name, param_distributions, X, y):
        model_lookup = {
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "ElasticNet": ElasticNet(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "SVR": SVR(),
            "K-Nearest Neighbors": KNeighborsRegressor()
        }
        model = model_lookup[model_name]
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_distributions,
            n_iter=50,
            cv=5,
            scoring='neg_mean_squared_error',
            random_state=42,
            n_jobs=-1
        )
        random_search.fit(X, y)
        return random_search
    return tune_model(model_name, _param_distributions, X, y)

if 'clear_cache' not in st.session_state:
    st.session_state.clear_cache = False

if st.session_state.clear_cache:
    st.cache_resource.clear()
    st.session_state.clear_cache = False
    st.success("Cached tuning results cleared.")

# ----------------------------- App Mode Selector -----------------------------
mode = st.radio("Choose Mode", ["Train a new model", "Tune hyperparameters", "Predict RT from saved model"])

# ----------------------------- Training Mode -----------------------------
if mode == "Train a new model":
    st.markdown("### 🧪 Step 1: Train a New Model")
    st.markdown("Use this section to train a machine learning model to predict HPLC retention times from molecular structures (SMILES).")
    with st.expander("**Instructions:**"):
        st.markdown("""
        1. Upload a CSV file with two columns:
            - `SMILES`: the chemical structure in SMILES format
            - `RT`: the experimental retention time
        2. Choose a regression model from the dropdown.
        3. Click the **Train Model** button.
        4. Once training completes, you can download the model and results.
        """)
    uploaded_file = st.file_uploader("Upload CSV with SMILES and RT", type="csv")
    if uploaded_file:
        input_data = pd.read_csv(uploaded_file)
        if not {'SMILES', 'RT'}.issubset(input_data.columns):
            st.error("CSV must contain 'SMILES' and 'RT' columns.")
            st.stop()

        smiles_list = input_data['SMILES'].to_list()
        rt_list = input_data['RT'].to_list()

        var_thresh = st.sidebar.slider("Variance threshold", 0.0, 1.0, 0.05)
        corr_thresh = st.sidebar.slider("Correlation threshold", 0.0, 1.0, 0.95)
        scale_features = st.sidebar.checkbox("Scale features", value=True)
        test_set_size = st.sidebar.slider("Test set size", 0.1, 0.9, 0.4)
        k = st.sidebar.slider("CV folds", 2, 10, 5)

        model_dict = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(max_iter=50000, tol=1e-3),
            "ElasticNet": ElasticNet(max_iter=50000, tol=1e-3),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "SVR": SVR(),
            "K-Nearest Neighbors": KNeighborsRegressor()
        }

        with st.spinner("Calculating descriptors..."):
            df, valid_rts = compute_training_descriptors(smiles_list, rt_list)
        if df.empty:
            st.error("No valid descriptors were generated.")
            st.stop()

        df = clean_descriptor_data(df, var_thresh, corr_thresh)
        data = pd.concat([df, pd.DataFrame({'RT': valid_rts})], axis=1)

        results, summary_df, fig = train_all_models_cached(data, model_dict, test_set_size, k, scale_features)

        st.pyplot(fig)

        with st.expander("Model Performance Summary"):
            st.dataframe(summary_df)

        all_results = []
        for name, res in results.items():
            df = res["results_df"].copy()
            df["Model"] = name
            df["Test MSE"] = res["test_mse"]
            all_results.append(df)

        combined_results_df = pd.concat(all_results, ignore_index=True)
        combined_csv = combined_results_df.to_csv(index=False).encode('utf-8')

        st.download_button("Download All Model Predictions", combined_csv, "all_model_results.csv", "text/csv")

        selected_model_name = st.selectbox("Select model to download parameters:", list(results.keys()))
        selected_model = results[selected_model_name]
        model_bundle = {
            'model': selected_model['model'],
            'features': selected_model['features'],
            'scaler': selected_model['scaler']
        }

        model_pickle = pickle.dumps(model_bundle)
        st.download_button(
            label=f"Download {selected_model_name} Model Parameters",
            data=model_pickle,
            file_name=f"{selected_model_name}_model.pkl",
            mime='application/octet-stream'
        )

# ----------------------------- Hyperparameter Tuning Mode -----------------------------
if mode == "Tune hyperparameters":
    st.markdown("### 📉 Step 2: Tune Model Hyperparameters")
    st.markdown("Improve model performance by optimizing its settings using randomized search cross-validation.")
    with st.expander("**Instructions:**"):
        st.markdown("""
        1. Upload the same type of dataset used in training (`SMILES`, `RT`).
        2. Select the model you want to tune.
        3. Click **Run Hyperparameter Tuning**.
        4. When tuning completes, review the best parameters and model score.
        5. Download the optimized model for use in predictions.
        """)
    uploaded_file = st.file_uploader("Upload CSV with SMILES and RT", type="csv")

    if uploaded_file:
        input_data = pd.read_csv(uploaded_file)
        if not {'SMILES', 'RT'}.issubset(input_data.columns):
            st.error("CSV must contain 'SMILES' and 'RT' columns.")
            st.stop()

        smiles_list = input_data['SMILES'].to_list()
        rt_list = input_data['RT'].to_list()

        var_thresh = st.sidebar.slider("Variance threshold", 0.0, 1.0, 0.05)
        corr_thresh = st.sidebar.slider("Correlation threshold", 0.0, 1.0, 0.95)
        scale_features = st.sidebar.checkbox("Scale features", value=True)

        model_choice = st.selectbox("Select model for hyperparameter tuning", [
            "Ridge", "Lasso", "ElasticNet", "Decision Tree", "Random Forest", "Gradient Boosting", "SVR", "K-Nearest Neighbors"])

        model_map = {
            "Ridge": (Ridge(), {"alpha": np.logspace(-4, 4, 100)}),
            "Lasso": (Lasso(max_iter=50000, tol=1e-3), {'alpha': np.logspace(-6, 2, 100)}),
            "ElasticNet": (ElasticNet(max_iter=50000, tol=1e-3), {'alpha': np.logspace(-6, 2, 100), "l1_ratio": np.linspace(0.01, 1.0, 50)}),
            "Decision Tree": (DecisionTreeRegressor(), {"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]}),
            "Random Forest": (RandomForestRegressor(), {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}),
            "Gradient Boosting": (GradientBoostingRegressor(), {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]}),
            "SVR": (SVR(), {"C": np.logspace(-2, 2, 10), "gamma": ['scale', 'auto']}),
            "K-Nearest Neighbors": (KNeighborsRegressor(), {"n_neighbors": list(range(1, 30))})
        }

        model, param_distributions = model_map[model_choice]

        with st.spinner("Calculating descriptors..."):
            df, valid_rts = compute_training_descriptors(smiles_list, rt_list)
        if df.empty:
            st.error("No valid descriptors were generated.")
            st.stop()

        df = clean_descriptor_data(df, var_thresh, corr_thresh)
        retention_times_df = pd.DataFrame({'RT': valid_rts})
        data = pd.concat([df, retention_times_df], axis=1)

        X = data.drop(columns=['RT'])
        y = data['RT']

        scaler = None
        if scale_features:
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        with st.spinner("Tuning hyperparameters..."):
            search = tune_model_cached(model_choice, param_distributions, X_train, y_train)

        best_model = search.best_estimator_
        st.success(f"Best Parameters: {search.best_params_}")

        y_pred = best_model.predict(X_test)
        y_train_pred = best_model.predict(X_train)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        st.write(f"**Performance on Test Set:**")
        st.write(f"- R²: {r2:.3f}")
        st.write(f"- MAE: {mae:.3f}")
        st.write(f"- MSE: {mse:.3f}")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_train, y_train_pred, alpha=0.6)
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        ax.set_xlabel("Experimental RT")
        ax.set_ylabel("Predicted RT")
        ax.set_title(f"{model_choice} - Predicted vs Experimental RT")
        st.pyplot(fig)

        bundle = {
            'model': best_model,
            'features': list(X.columns),
            'scaler': scaler,
            'params': search.best_params_
        }
        model_pickle = pickle.dumps(bundle)

        st.download_button(
            label=f"Download Tuned {model_choice} Model Parameters",
            data=model_pickle,
            file_name=f"{model_choice}_tuned_model.pkl",
            mime='application/octet-stream'
        )

# ----------------------------- Prediction Mode -----------------------------
elif mode == "Predict RT from saved model":
    st.markdown("### 🔍 Step 3: Predict Retention Times for New Compounds")
    st.markdown("Use a trained model to predict retention times for unknown compounds.")
    with st.expander("**Instructions:**"):
        st.markdown("""
        1. Upload a CSV file with a column `SMILES` (no `RT` column needed).
        2. Upload a trained model file (`.pkl`) from the training or tuning steps.
        3. Click **Predict** to see predicted retention times.
        4. Download the predictions as a CSV file.
        """)
    model_file = st.file_uploader("Upload saved model (.pkl)", type="pkl")
    smiles_file = st.file_uploader("Upload CSV with SMILES", type="csv")

    if model_file and smiles_file:
        model_bundle = pickle.load(model_file)
        input_smiles_df = pd.read_csv(smiles_file)

        if 'SMILES' not in input_smiles_df.columns:
            st.error("CSV must contain a 'SMILES' column.")
            st.stop()

        smiles_list = input_smiles_df['SMILES'].to_list()
        with st.spinner("Calculating descriptors..."):
            df = compute_prediction_descriptors(smiles_list)

        if df.empty:
            st.error("No valid descriptors were generated.")
            st.stop()

        df = df[model_bundle['features']]
        if model_bundle['scaler']:
            df = pd.DataFrame(model_bundle['scaler'].transform(df), columns=df.columns)

        preds = model_bundle['model'].predict(df)
        results = pd.DataFrame({'SMILES': smiles_list, 'Predicted RT': preds})
        st.write(results)

        pred_csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", pred_csv, "predicted_rt.csv", "text/csv")
