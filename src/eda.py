import pandas as pd

def get_distributions(df):
    """
    Extracts distribution data for both numerical and categorical columns
    to be easily visualized in Streamlit (e.g., histograms or value counts).
    """
    distributions = {}
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            # For numerical columns, we can return the series values for plotting histograms
            distributions[col] = {"type": "numerical", "data": df[col].dropna()}
        else:
            # For categorical columns, we return value counts for bar charts
            distributions[col] = {"type": "categorical", "data": df[col].value_counts()}

    return distributions

def get_correlation_matrix(df):
    """
    Computes the correlation matrix for numerical features to highlight relationships.
    """
    num_df = df.select_dtypes(include=['int64', 'float64'])

    if not num_df.empty and num_df.shape[1] > 1:
        return num_df.corr()
    return None
