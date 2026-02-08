import os
import json
import time
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
api_key=os.getenv("API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0,
    api_key=api_key,
    max_retries=2, # Automatically retry on failure
)

def convert_text_to_df(raw_text):
    """
    Uses Llama 3.3 on Groq to parse messy bank text into a structured DataFrame.
    """
    
    prompt_template = """
    You are a specialized financial data extraction AI.
    I will give you raw text from a bank statement. Your job is to extract transaction details.
    
    Rules:
    1. Extract specific transactions. Look for dates, descriptions, and amounts.
    2. Ignore "Opening Balance", "Closing Balance", and generic headers.
    3. Identify if the amount is a "Credit" (Deposit) or "Debit" (Withdrawal). 
       - If only one amount column exists, assume it is a Debit unless it says "Received" or "Deposit".
    4. Output ONLY a valid JSON list of objects. No markdown formatting.
    
    Format:
    [
        {{"Date": "DD-MM-YYYY", "Description": "Transaction Name", "Amount": 100.00, "Type": "Debit"}},
        ...
    ]
    
    Here is the text:
    {text}
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = prompt | llm
    
    try:
        # Llama 3.3 has a 128k context window, so 6000 chars is easy for it.
        response = chain.invoke({"text": raw_text[:8000]})
        
        # Clean the response 
        json_str = response.content.replace("```json", "").replace("```", "").strip()
        
        data = json.loads(json_str)
        return pd.DataFrame(data)
        
    except Exception as e:
        print(f"Error in extraction: {e}")
        return None