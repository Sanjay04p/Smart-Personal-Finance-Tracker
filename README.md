# 💳 Smart Personal Finance Tracker

An AI-powered financial dashboard that transforms raw bank statements (PDF/CSV) into actionable insights using GenAI and Machine Learning.

## 🚀 Features
* **Multi-Format Ingestion:** Extracts data from complex PDF bank statements using **Llama 3.3**.
* **Smart Categorization:** Classifies transactions (e.g., "Uber" -> "Transport") using **Llama 3.3** with an intelligent caching layer to reduce API costs.
* **Spending Forecast:** Predicts month-end spending using **Linear Regression (Scikit-Learn)** based on daily spending velocity.
* **Chat with Data:** RAG-style Q&A interface to ask questions like *"How much did I spend on food?"*.

## 🛠️ Tech Stack
* **Frontend:** Streamlit, Plotly
* **AI Models:** Llama 3.3 via Groq (Extraction,Reasoning)
* **Machine Learning:** Scikit-Learn (Forecasting)
* **Data Processing:** Pandas, PDFPlumber

## ⚙️ Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Set up `.env` with `GROQ_API_KEY` 
4. Run the app: `streamlit run app.py`

## Live Demo

https://github.com/user-attachments/assets/c9ee4714-fc04-4db7-83e7-6a1c1366cb91




