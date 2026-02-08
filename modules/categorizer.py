import pandas as pd
import json
import streamlit as st 
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()
api_key=os.getenv("API_KEY")
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0,
    api_key=api_key
)

def categorize_transactions(df):
    """
    Categorizes transaction descriptions using Llama 4.
    """
    
    # 1. Identify the 'Description' column dynamically
    # We look for common names or just pick the 2nd column (usually description)
    possible_names = ['Description', 'Transactiondetails', 'Transaction Details', 'Particulars', 'Narration']
    desc_col = next((col for col in df.columns if col in possible_names), None)
    
    # If we can't find a name match, we guess it's the column with the most unique string values
    if not desc_col:
        # Heuristic: Description columns usually have high cardinality and are object type
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
             # Assume the longest text column is the description
             desc_col = object_cols[df[object_cols].apply(lambda x: x.str.len().mean()).argmax()]
    
    if not desc_col:
        st.error("Could not find a 'Description' column to categorize.")
        return df

    # 2. Get unique descriptions
    unique_desc = df[desc_col].unique().tolist()
    
    # DEBUG: Show user what we are sending (Optional, good for troubleshooting)
    # st.write(f"Categorizing based on column: **{desc_col}**")
    
    prompt_template = """
    You are a financial classifier. Map the following bank transaction descriptions to these categories:
    [Food, Transport, Shopping, Bills, Entertainment, Health, Transfer, Income, Other]
    
    Rules:
    - Return ONLY a valid JSON object where keys are the descriptions and values are the categories.
    - No markdown formatting.
    
    Descriptions:
    {descriptions}
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["descriptions"])
    chain = prompt | llm
    
    try:
        response = chain.invoke({"descriptions": str(unique_desc)})
        
        json_str = response.content.replace("```json", "").replace("```", "").strip()
        category_map = json.loads(json_str)
        
        # 3. Map back using the detected column name
        df['Category'] = df[desc_col].map(category_map)
        df['Category'] = df['Category'].fillna('Other')
        
        return df
        
    except Exception as e:
        st.error(f"Categorization Error: {e}") 
        return df