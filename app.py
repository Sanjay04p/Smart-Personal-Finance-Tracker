import streamlit as st
import pandas as pd
import plotly.express as px

# Import our modules
from modules.parser import load_data
from modules.llm_extractor import convert_text_to_df
from modules.categorizer import categorize_transactions
from modules.forecaster import predict_month_end
from modules.chat import process_query

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Smart Finance Tracker", 
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR MODERN LOOK ---
st.markdown("""
    <style>
    .main { padding-top: 2rem; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("💳 Smart Personal Finance Tracker")
st.markdown("Your AI-powered financial assistant.")

# --- INITIALIZE SESSION STATE ---
if 'data' not in st.session_state:
    st.session_state['data'] = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Data Source")
    uploaded_file = st.file_uploader("Upload Statement (PDF/CSV)", type=["csv", "pdf"])
    
    st.markdown("---")
    if st.button("🗑️ Clear / Reset App", use_container_width=True):
        st.session_state['data'] = None
        st.rerun()
        
    st.markdown("### ℹ️ How to use")
    st.info("1. Upload your bank statement.\n2. Go to 'Data' tab to Categorize.\n3. Explore Insights & Chat!")

# --- MAIN LOGIC ---
if uploaded_file is not None:
    
    # 1. LOAD DATA (One time only)
    if st.session_state['data'] is None:
        # Check file type and load
        if uploaded_file.name.endswith(".csv"):
            st.session_state['data'] = load_data(uploaded_file)
        else:
            with st.spinner("🤖 AI is reading your statement..."):
                raw_text = load_data(uploaded_file)
                st.session_state['data'] = convert_text_to_df(raw_text)
                
    # 2. DATA NORMALIZATION (Critical Logic)
    if st.session_state['data'] is not None and not st.session_state['data'].empty:
        df = st.session_state['data'].copy()
        
        # Standardize Columns
        rename_map = {
            'Date&time': 'Date', 'Transactiondetails': 'Description', 
            'Transaction Details': 'Description', 'Particulars': 'Description'
        }
        df.rename(columns=rename_map, inplace=True)
        
        # Clean Amount
        if 'Amount' in df.columns:
            df['Amount'] = df['Amount'].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        
        # Save cleaned data
        st.session_state['data'] = df
        
        # --- TOP LEVEL METRICS (DASHBOARD) ---
        # Calculate quick stats if Category exists
        if 'Category' in df.columns:
            income = df[df['Category'] == 'Income']['Amount'].sum()
            expense = df[~df['Category'].isin(['Income', 'Transfer', 'Other'])]['Amount'].sum()
            balance = income - expense
            
            # 3-Column Layout for Key Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("💰 Total Income", f"₹{income:,.2f}")
            col2.metric("💸 Total Spending", f"₹{expense:,.2f}", delta=f"-{(expense/income)*100:.1f}%" if income > 0 else None)
            col3.metric("🏦 Remaining Balance", f"₹{balance:,.2f}")
            st.markdown("---")

        # --- TABS LAYOUT ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "📄 Data & Categorization", 
            "📊 Visualization", 
            "🔮 Forecasting", 
            "💬 Chat with AI"
        ])

        # --- TAB 1: DATA ---
        with tab1:
            st.subheader("Transaction Log")
            
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.dataframe(df, use_container_width=True, height=400)
            with col_b:
                st.markdown("### Actions")
                if st.button("🏷️ Run AI Categorization", use_container_width=True, type="primary"):
                    with st.spinner("Classifying transactions..."):
                        st.session_state['data'] = categorize_transactions(st.session_state['data'])
                        st.success("Done!")
                        st.rerun()
                st.caption("Click to label transactions like 'Food', 'Travel' automatically.")

        # --- TAB 2: VISUALIZATION ---
        with tab2:
            st.subheader("Spending Insights")
            
            if 'Category' in df.columns:
                # Filter Logic
                spending_df = df[~df['Category'].isin(['Income', 'Transfer', 'Deposit'])].copy()
                spending_df = spending_df.dropna(subset=['Amount'])
                
                if not spending_df.empty:
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        st.markdown("#### Category Breakdown")
                        fig_pie = px.pie(spending_df, names='Category', values='Amount', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
                    with c2:
                        st.markdown("#### Top Expenses")
                        # Sort by amount and show top 10
                        top_expenses = spending_df.sort_values(by='Amount', ascending=False).head(10)
                        st.dataframe(top_expenses[['Date', 'Description', 'Amount', 'Category']], hide_index=True)
                else:
                    st.info("No spending data found. Try categorizing your data first.")
            else:
                st.warning("⚠️ Please run categorization in Tab 1 first.")

        # --- TAB 3: FORECASTING ---
        with tab3:
            st.subheader("Month-End Projection")
            st.caption("Based on your current daily spending velocity.")
            
            if 'Category' in df.columns:
                if st.button("🔮 Generate Forecast"):
                    with st.spinner("Analyzing spending trends..."):
                        # Prepare spending data
                        spending_df = df[~df['Category'].isin(['Income', 'Transfer', 'Other'])].copy()
                        
                        try:
                            result = predict_month_end(spending_df)
                            if result and result[0] is not None:
                                fig_forecast, predicted_total = result
                                current_total = spending_df['Amount'].sum()
                                
                                m1, m2 = st.columns(2)
                                m1.metric("Current Spend", f"₹{current_total:,.2f}")
                                m2.metric("Predicted Month-End", f"₹{predicted_total:,.2f}")
                                
                                st.plotly_chart(fig_forecast, use_container_width=True)
                            else:
                                st.warning("Not enough data to forecast (need at least 2 days of activity).")
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                st.warning("⚠️ Please run categorization in Tab 1 first.")

        # --- TAB 4: CHAT ---
        with tab4:
            st.subheader("Chat with your Finances")
            
            # Simple chat interface
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # React to user input
            if prompt := st.chat_input("Ask something like 'How much did I spend on food? '"):
                # Display user message
                st.chat_message("user").markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})

                with st.spinner("Thinking..."):
                    response = process_query(df, prompt)
                
                # Display assistant response
                st.chat_message("assistant").markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    else:
        st.error("Data extraction failed. Please try a different file.")

else:
    # Landing Page content when no file is uploaded
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 50px;'>
        <h3>👋 Welcome to Smart Finance Tracker</h3>
        <p>Upload a Bank PDF or CSV to get started.</p>
    </div>
    """, unsafe_allow_html=True)