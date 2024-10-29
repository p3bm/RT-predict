import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, Crippen, rdMolDescriptors
from rdkit import RDLogger
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np
import matplotlib.pyplot as plt
import pickle

RDLogger.DisableLog('rdApp.*')

st.title("HPLC Retention Time Prediction")

def compute_3d_descriptors(mol):
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    return Descriptors3D.CalcMolDescriptors3D(mol)

def compute_2d_descriptors(mol):
    return Descriptors.CalcMolDescriptors(mol)

def compute_additional_descriptors(mol):
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)

def descriptors_from_smiles(smiles_list):
    data = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol = Chem.AddHs(mol)
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                raise ValueError(f"Sanitization failed: {e}")
            
            descriptors_3d = compute_3d_descriptors(mol)
            descriptors_2d = compute_2d_descriptors(mol)
            
            descriptors = {**descriptors_2d, **descriptors_3d}
            data.append(descriptors)
        else:
            st.warning(f"Invalid SMILES string: {smiles}")
    
    df = pd.DataFrame(data)
    df.replace("", float("NaN"), inplace=True)
    df.dropna(how='all', axis=1, inplace=True)
    
    st.write(f"Total no. of descriptors: {df.shape[1]}")
    return df

def clean_descriptor_data(df, var_thresh, corr_thresh):
    var_vals = df.var(numeric_only=True)
    for descriptor, variance in var_vals.items():
        if variance < var_thresh:
            df.drop(descriptor, axis=1, inplace=True)
    
    st.write(f"No. of descriptors after low var removal: {df.shape[1]}")
    
    corr_vals = df.corr(numeric_only=True)
    for descriptor_x, slice in corr_vals.items():
        for descriptor_y, correlation in slice.items():
            if descriptor_x != descriptor_y:
                if correlation > corr_thresh:
                    try:
                        df.drop(descriptor_x, axis=1, inplace=True)
                    except KeyError:
                        pass
    
    st.write(f"No. of descriptors after high corr removal: {df.shape[1]}")
    return df

def train_model(data, model_type, test_set_size, k):
    X = data.drop(columns=['RT'])
    y = data['RT']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size)
    model = model_type
    mse_scorer = make_scorer(mean_squared_error)
    cv_scores = cross_val_score(model, X_train, y_train, cv=k, scoring=mse_scorer)
    
    st.write(f"CV scores: {cv_scores}")
    st.write(f"Mean cross-validation MSE: {np.mean(cv_scores)}")
    
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    st.write(f"Test set MSE: {test_mse}")
    
    y_train_pred = model.predict(X_train)
    plt.scatter(y_test, y_test_pred, label="Test")
    plt.scatter(y_train, y_train_pred, label="Train")
    plt.xlabel("Experimental RT (mins)")
    plt.ylabel("Predicted RT (mins)")
    plt.legend()

    y_test = y_test.tolist()
    y_test_pred = y_test_pred.tolist()

    for i in range(0,len(y_test)):
        for idx, row in input_data.iterrows():
            if row['RT'] == y_test[i]:
                plt.annotate(row['Compound'],(y_test[i],y_test_pred[i]))
    
    y_equals_x = np.linspace(0, 4)
    plt.plot(y_equals_x, y_equals_x, 'r--')
    st.pyplot(plt)
    
    results_df = pd.DataFrame({'Experimental RT': y_test, 'Predicted RT': y_test_pred})

    return model, results_df

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    input_data = pd.read_csv(uploaded_file)
    
    smiles_list = input_data['SMILES'].to_list()
    rt_list = input_data['RT'].to_list()
    
    var_thresh = st.sidebar.slider("Variance threshold", 0.0, 1.0, 0.05)
    corr_thresh = st.sidebar.slider("Correlation threshold", 0.0, 1.0, 0.95)
    
    model_type = st.sidebar.selectbox("Model Type", 
                                      ["Linear Regression", "Ridge", "Lasso", "ElasticNet", 
                                       "Decision Tree", "Random Forest", "Gradient Boosting", 
                                       "SVR", "K-Nearest Neighbors"])
    
    test_set_size = st.sidebar.slider("Test set size", 0.1, 0.9, 0.4)
    k = st.sidebar.slider("Number of CV folds", 2, 10, 5)
    
    model_dict = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "SVR": SVR(),
        "K-Nearest Neighbors": KNeighborsRegressor()
    }
    
    model = model_dict[model_type]
    
    df = descriptors_from_smiles(smiles_list)
    df = clean_descriptor_data(df, var_thresh, corr_thresh)
    
    retention_times_df = pd.DataFrame({'RT': rt_list})
    data = pd.concat([df, retention_times_df], axis=1)
    
    trained_model, results_df = train_model(data, model, test_set_size, k)
    
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Results", data=csv, file_name='results.csv', mime='text/csv')
    #st.download_button(label="Download model", data=joblib_file, file_name='model_params.csv', mime='text/plain')

    # Save the model to a pickle file
    model_pickle = pickle.dumps(trained_model)
    st.download_button(label="Download Model Parameters", data=model_pickle, file_name='model_parameters.pkl', mime='application/octet-stream')