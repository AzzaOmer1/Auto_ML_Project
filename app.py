# app.py
import streamlit as st
import pandas as pd
import pickle
from config import Config
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_data
from src.preprocessor import preprocess_data
from src.model_selector import train_and_evaluate_models, tune_and_evaluate_best_model
from src.eda import get_distributions, get_correlation_matrix

# Streamlit page configuration
st.set_page_config(
    page_title="Professional AutoML & MLOps System",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Professional AutoML Platform with MLflow")
st.markdown("Build, preprocess, train, and track high-performance Machine Learning models seamlessly.")

# --- SIDEBAR: Configuration & Inputs ---
st.sidebar.header("1. Data & Settings")

uploaded_file = st.sidebar.file_uploader("Upload your Dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load Data
    try:
        df = load_data(uploaded_file)
        st.sidebar.success("Dataset loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        st.stop()

    # Main Area: Dataset Preview
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Dataset Basic Info
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", df.isna().sum().sum())

    # Select Target Column
    st.sidebar.header("2. Target & Problem Setup")
    target_column = st.sidebar.selectbox("Select Target Column", options=df.columns)

    # Auto-detect problem type based on target dtype or unique values
    is_numeric_target = pd.api.types.is_numeric_dtype(df[target_column])
    unique_vals = df[target_column].nunique()

    if not is_numeric_target or unique_vals < 10:
        default_problem_type = "Classification"
    else:
        default_problem_type = "Regression"

    problem_type = st.sidebar.selectbox(
        "Select Problem Type",
        options=["Classification", "Regression"],
        index=0 if default_problem_type == "Classification" else 1
    )

    # Model Selection from Config options
    st.sidebar.header("3. Model Selection")
    if problem_type == "Classification":
        available_models = ["Logistic Regression", "Random Forest Classifier", "XGBoost", "LightGBM"]
    else:
        available_models = ["Linear Regression", "Ridge", "Random Forest Regressor", "XGBoost Regressor", "LightGBM Regressor"]

    selected_models = st.sidebar.multiselect("Choose Models to Train", options=available_models, default=available_models[:2])


    # --- Exploratory Data Analysis (EDA) ---
    with st.expander("📊 Exploratory Data Analysis (EDA)", expanded=False):
        st.subheader("Dataset Summary & Statistics")
        st.dataframe(df.describe(), use_container_width=True)

        # Call separate distributions function
        distributions = get_distributions(df)

        st.subheader("Feature Distribution")
        # Extract numerical columns from the dictionary returned by the function
        numeric_columns = [col for col, info in distributions.items() if info["type"] == "numerical"]

        if numeric_columns:
            selected_col = st.selectbox("Select numeric column for Distribution plot", numeric_columns)

            fig, ax = plt.subplots(figsize=(8, 4))
            # Use pre-processed data from the separate function
            sns.histplot(distributions[selected_col]["data"], kde=True, ax=ax, color="skyblue")
            ax.set_title(f"Distribution of {selected_col}")
            st.pyplot(fig)
        else:
            st.info("No numeric columns available for distribution plot.")

        # Call separate correlation matrix function
        st.subheader("Correlation Heatmap")
        corr_matrix = get_correlation_matrix(df)

        if corr_matrix is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Correlation Matrix")
            st.pyplot(fig)
        else:
            st.info("Not enough numeric columns for correlation heatmap.")

    # --- Preprocessing & Training Trigger ---
    if st.sidebar.button("Run Preprocessing & Train Models", type="primary"):
        if not selected_models:
            st.warning("Please select at least one model to train.")
        else:
            with st.spinner("Preprocessing data with strict 3-way split..."):
                try:
                    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df, target_column)
                    # Store data in session_state to prevent loss during interaction
                    st.session_state['X_train'] = X_train
                    st.session_state['X_val'] = X_val
                    st.session_state['X_test'] = X_test
                    st.session_state['y_train'] = y_train
                    st.session_state['y_val'] = y_val
                    st.session_state['y_test'] = y_test
                    st.success("Preprocessing completed successfully!")

                    st.info(f"Train set shape: {X_train.shape} | Validation set shape: {X_val.shape} | Test set shape: {X_test.shape}")
                except Exception as e:
                    st.error(f"Error during preprocessing: {e}")
                    st.stop()

            with st.spinner("Training models and logging experiments to MLflow..."):
                try:
                    results_df, trained_models = train_and_evaluate_models(
                        X_train, X_val, y_train, y_val, selected_models, problem_type
                    )
                    # Store results in session_state to persist them in the UI
                    st.session_state['results_df'] = results_df
                    st.session_state['trained_models'] = trained_models
                except Exception as e:
                    st.error(f"Error during model training: {e}")

    # Display results if stored in session_state
    if 'results_df' in st.session_state and not st.session_state['results_df'].empty:
        st.subheader("🏆 Model Evaluation Results (Validation Set)")
        st.dataframe(st.session_state['results_df'], use_container_width=True)
        st.success(f"Experiments successfully logged to MLflow under experiment: **{Config.MLFLOW_EXPERIMENT_NAME}**")

        # --- Professional Stage: Automated Tuning for All Selected Modelsالمرحلة الاحترافية: تطبيق الـ   ---
        st.markdown("---")
        st.subheader("🌟 Automated Hyperparameter Tuning & Test Evaluation")
        st.info("The system will automatically optimize hyperparameters for all models you selected and evaluate them on the Test Set.")

        if st.button("🚀 Run Automated Tuning & Final Evaluation for All Selected Models", type="primary"):
            with st.spinner("Optimizing hyperparameters for all selected models... Please wait."):
                try:
                    tuned_results = []
                    best_tuned_score = -float('inf')
                    best_model_obj = None
                    best_model_name_final = ""

                    # Iterate through all selected models and perform automated tuningالمرور على كل النماذج التي تم اختيارها وتطبيق الـ Tuning عليها تلقائياً
                    for model_name in selected_models:
                        final_model, test_metrics = tune_and_evaluate_best_model(
                            st.session_state['X_train'], st.session_state['X_val'], st.session_state['X_test'],
                            st.session_state['y_train'], st.session_state['y_val'], st.session_state['y_test'],
                            model_name, problem_type
                        )

                        metrics_row = {"Model": model_name, **test_metrics}
                        tuned_results.append(metrics_row)

                        # ---Smart Correction to Select the Ultimate Best Model ---
                        if problem_type == "Regression":
                            # In regression, the best is the one with the highest R2-Score (%)
                            score_val = test_metrics.get("R2-Score (%)", list(test_metrics.values())[0])
                        else:
                            # In classification, take the primary performance metric (e.g., Accuracy or F1)
                            score_val = list(test_metrics.values())[0]

                        if score_val > best_tuned_score:
                            best_tuned_score = score_val
                            best_model_obj = final_model
                            best_model_name_final = model_name

                    st.success("Automated Tuning & Test Evaluation Completed Successfully!")

                    # Display model comparison table after tuning on the Test Set
                    st.subheader("🎯 Test Set Performance Metrics (After Tuning)")
                    tuned_df = pd.DataFrame(tuned_results)
                    st.dataframe(tuned_df, use_container_width=True)

                    # Highlight the ultimate best model and provide direct download button based on highest true efficiency
                    st.markdown(f"### 🏆 Best Model Identified: **{best_model_name_final}**")

                    model_bytes = pickle.dumps(best_model_obj)
                    st.download_button(
                        label=f"📥 Download Best Production-Ready Model ({best_model_name_final})",
                        data=model_bytes,
                        file_name=f"production_{best_model_name_final.lower().replace(' ', '_')}.pkl",
                        mime="application/octet-stream"
                    )

                except Exception as e:
                    st.error(f"Error during automated tuning: {e}")

else:
    st.info("👈 Please upload a dataset from the sidebar to get started.")
