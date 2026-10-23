import pandas as pd

def load_data(file_path):
    """
    Unified and dynamically designed function to read data files (CSV or Excel).
    Inputs: file_path (The uploaded file via Streamlit interface)
    Outputs: DataFrame ready for subsequent processing steps
    """
    try:
        # Check if the uploaded file extension is CSV
        if file_path.name.endswith('.csv'):
            df = pd.read_csv(file_path)

        # Check if the file is an Excel format (old or modern)
        elif file_path.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)

        # Raise an error programmatically to prevent system crash if format is unsupported
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")

        # Return the successfully loaded dataset
        return df

    except Exception as e:
        # Catch any unexpected error during loading and raise a clear descriptive message
        raise Exception(f"An error occurred while reading the file: {e}")
