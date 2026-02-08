import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()
api_key=os.getenv("API_KEY")
# Using Llama 3.3 for its reasoning capabilities and large context window
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=api_key)

def process_query(df, query):
    """
    Takes the financial dataframe and a user query, 
    and returns an AI-generated answer.
    """
    # 1. Prepare Data
    # Convert DataFrame to a string format (CSV-like) so the LLM can read it
    # We select relevant columns to save tokens and reduce noise
    relevant_cols = ['Date', 'Description', 'Amount', 'Category']
    
    # Ensure these columns exist
    available_cols = [c for c in relevant_cols if c in df.columns]
    data_str = df[available_cols].to_string(index=False)
    
    # 2. Define the Prompt
    prompt_template = """
    You are an expert personal finance analyst. 
    Analyze the following bank statement data and answer the user's question accurately.
    
    Data Context:
    {data}
    
    User Question: {query}
    
    Guidelines:
    - Be concise and direct.
    - If calculating totals, double-check your math.
    - If the answer isn't in the data, say "I cannot find that information."
    - Format currency as ₹ (Rupees).
    
    Answer:
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["data", "query"])
    chain = prompt | llm
    
    try:
        response = chain.invoke({
            "data": data_str,
            "query": query
        })
        return response.content
        
    except Exception as e:
        return f"Error processing your question: {e}"