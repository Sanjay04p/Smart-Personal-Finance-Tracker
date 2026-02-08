import pandas as pd
import pdfplumber
import io

def load_data(uploaded_file):
    """
    Main entry point to load data based on file extension.
    """
    if uploaded_file.name.endswith('.csv'):
        return load_csv(uploaded_file)
    elif uploaded_file.name.endswith('.pdf'):
        return load_pdf_text(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Please upload CSV or PDF.")

def load_csv(uploaded_file):
    """Simple CSV loader."""
    try:
        # Bank CSVs sometimes have meta-data in top rows, so we might need to skip
        # For now, we assume a standard header exists.
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        return f"Error reading CSV: {e}"

def load_pdf_text(uploaded_file):
    """
    Extracts RAW TEXT from the PDF. We will let the LLM structure it later.
    """
    full_text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"
    
    return full_text