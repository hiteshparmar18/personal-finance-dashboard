# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO, StringIO
from dateutil import parser
import re
import difflib
import json

st.set_page_config(page_title="Personal Finance Dashboard", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Styling (Finance Green Theme)
# -------------------------
GREEN = "#057a55"
LIGHT_GREEN = "#eaf6ef"
ACCENT = "#0b8f67"
st.markdown(
    f"""
    <style>
    /* Page background */
    .reportview-container .main {{
        background-color: #ffffff;
    }}
    /* Header */
    .css-1d391kg {{
        color: {GREEN};
    }}
    /* Sidebar */
    .css-1d391kg + div .css-1lcbmhc {{
        background-color: {LIGHT_GREEN};
    }}
    /* Metric labels */
    .metric-label {{
        color: #333333 !important;
    }}
    /* Card like containers */
    .card {{
        background: linear-gradient(180deg, rgba(250,250,250,0.9), rgba(245,255,250,0.9));
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }}
    /* Make buttons green */
    .stButton>button, .stDownloadButton>button {{
        background-color: {ACCENT} !important;
        color: white !important;
    }}
    /* Table header */
    .stDataFrame thead th {{
        background-color: {GREEN} !important;
        color: white !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Helper / Categorizer
# -------------------------
DEFAULT_CATEGORY_KEYWORDS = {
    "Food & Dining": ["restaurant", "cafe", "mcdonald", "starbucks", "domino", "ubereats", "zomato", "food", "canteen", "dining", "swiggy", "pizza"],
    "Groceries": ["supermarket", "grocery", "grocer", "bigbasket", "reliance", "dmart", "spencer", "grocery"],
    "Transport": ["uber", "ola", "taxi", "fuel", "petrol", "bus", "metro", "flight", "irctc", "train", "auto"],
    "Shopping": ["amazon", "flipkart", "shopping", "myntra", "shop", "mall"],
    "Bills & Utilities": ["electricity", "water", "gas", "utility", "bill", "netflix", "amazonprime", "spotify", "phone", "internet"],
    "Salary": ["salary", "payroll", "credited", "ctc", "pay"],
    "Rent": ["rent", "house rent"],
    "Health": ["clinic", "hospital", "pharmacy", "medical", "doctor"],
    "Entertainment": ["movie", "cinema", "theatre", "concert", "games"],
    "Transfer": ["transfer", "neft", "imps", "rtgs", "to", "from", "received", "deposit"],
    "Others": []
}

def detect_columns(df):
    """
    Try to detect date/description/amount columns heuristically.
    Returns (date_col, desc_col, amount_col, debit_credit_col_or_type_col)
    """
    colnames = [c.lower() for c in df.columns]
    date_col = None
    desc_col = None
    amount_col = None
    type_col = None

    for c in df.columns:
        lc = c.lower()
        if 'date' in lc or 'transaction date' in lc or 'txn date' in lc or 'value date' in lc:
            date_col = c
        if any(k in lc for k in ['description','details','narration','particular','remarks','description','trans_desc','merchant']):
            desc_col = c
        if any(k in lc for k in ['amount','amt','value','withdrawal','debit','credit','amount (in rs)']):
            # choose the best candidate later; handle positive/negative amounts
            amount_col = c if amount_col is None else amount_col
        if any(k in lc for k in ['type','credit','debit','dr','cr','txn type']):
            type_col = c

    # fallback heuristics
    if date_col is None:
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                date_col = c
                break

    if desc_col is None:
        for c in df.columns:
            if df[c].dtype == object:
                desc_col = c
                break

    if amount_col is None:
        # pick numeric column that isn't the date
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]) and c != date_col:
                amount_col = c
                break

    return date_col, desc_col, amount_col, type_col

def clean_and_normalize(df):
    # Detect columns
    date_col, desc_col, amount_col, type_col = detect_columns(df)
    if date_col is None or desc_col is None or amount_col is None:
        raise ValueError("Could not detect Date/Description/Amount columns automatically. Please ensure your file has Date, Description, and Amount columns or provide a cleaned CSV.")

    # Date parsing
    df = df.copy()
    try:
        df['Date'] = pd.to_datetime(df[date_col])
    except Exception:
        df['Date'] = df[date_col].apply(lambda x: parser.parse(str(x)) if pd.notnull(x) else x)

    # Description
    df['Description'] = df[desc_col].astype(str).str.strip()

    # Amount: try to derive a single signed 'Amount' column
    amt = df[amount_col]
    if type_col is not None:
        types = df[type_col].astype(str).str.lower()
        signed = []
        for a, t in zip(amt, types):
            try:
                val = float(str(a).replace(',','')) if pd.notnull(a) and str(a).strip() != '' else 0.0
            except:
                val = 0.0
            if any(k in t for k in ['dr','debit','withdrawal','withdraw']):
                signed.append(-abs(val))
            elif any(k in t for k in ['cr','credit','deposit','received']):
                signed.append(abs(val))
            else:
                signed.append(val)
        df['Amount'] = signed
    else:
        # If there are separate Debit/Credit or Withdrawal/Deposit columns, try to find them
        lowcols = [c.lower() for c in df.columns]
        if 'withdrawal' in lowcols and 'deposit' in lowcols:
            wcol = df.columns[lowcols.index('withdrawal')]
            dcol = df.columns[lowcols.index('deposit')]
            df['Amount'] = pd.to_numeric(df[dcol].fillna(0), errors='coerce').fillna(0) - pd.to_numeric(df[wcol].fillna(0), errors='coerce').fillna(0)
        elif 'debit' in lowcols and 'credit' in lowcols:
            wcol = df.columns[lowcols.index('debit')]
            dcol = df.columns[lowcols.index('credit')]
            df['Amount'] = pd.to_numeric(df[dcol].fillna(0), errors='coerce').fillna(0) - pd.to_numeric(df[wcol].fillna(0), errors='coerce').fillna(0)
        else:
            # fallback: try to parse numeric and leave sign as is; if all amounts are positive but there's a Debit column,
            # we attempted earlier; otherwise assume negative = expense
            df['Amount'] = pd.to_numeric(amt.astype(str).str.replace(',',''), errors='coerce').fillna(0)

    # Category placeholder
    df['Category'] = "Uncategorized"

    # Month and Year helpers
    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    return df[['Date','Description','Amount','Category','YearMonth','Month','Year']]

def keyword_category(description, keyword_map):
    desc = description.lower()
    for cat, keywords in keyword_map.items():
        for kw in keywords:
            if kw in desc:
                return cat
    return None

def fuzzy_category(description, keyword_map):
    desc = description.lower()
    all_keywords = []
    mapping = {}
    for cat, keywords in keyword_map.items():
        for kw in keywords:
            all_keywords.append(kw)
            mapping[kw] = cat
    tokens = re.split(r'\W+', desc)
    for t in tokens:
        matches = difflib.get_close_matches(t, all_keywords, n=1, cutoff=0.82)
        if matches:
            return mapping[matches[0]]
    return None

def categorize_transactions(df, custom_map=None):
    keyword_map = DEFAULT_CATEGORY_KEYWORDS.copy()
    if custom_map:
        # merge custom map
        for k, v in custom_map.items():
            if isinstance(v, list):
                keyword_map.setdefault(k, []).extend(v)
            elif isinstance(v, str):
                keyword_map.setdefault(k, []).append(v)
    cats = []
    for desc in df['Description'].fillna(''):
        cat = keyword_category(desc, keyword_map)
        if cat is None:
            cat = fuzzy_category(desc, keyword_map)
        if cat is None:
            cat = 'Others'
        cats.append(cat)
    df['Category'] = cats
    return df

# -------------------------
# Sample CSV generator
# -------------------------
SAMPLE_CSV = """Date,Description,Amount
2025-01-03,STARBUCKS COFFEE,-4.50
2025-01-05,SALARY - ACME CORP,2500.00
2025-01-06,GROCERY MART,-50.00
2025-01-07,UBER TRIP,-12.00
2025-01-11,UBER TRIP,-7.20
2025-01-13,NETFLIX SUBSCRIPTION,-15.99
2025-01-15,DMART STORE,-35.00
2025-01-17,RECEIVED TRANSFER,150.00
2025-01-20,PHARMACY SHOP,-20.00
2025-01-25,AMAZON MARKETPLACE,-60.00
2025-02-02,RENT PAYMENT,-600.00
2025-02-05,STARBUCKS COFFEE,-5.00
2025-02-07,UBER TRIP,-9.50
2025-02-10,AMAZON MARKETPLACE,-45.00
2025-02-12,SALARY - ACME CORP,2500.00
2025-02-15,DMART STORE,-40.00
2025-02-18,RECEIVED TRANSFER,200.00
2025-02-20,PHARMACY SHOP,-30.00
2025-02-25,FLIPKART ONLINE,-25.00
2025-03-01,STARBUCKS COFFEE,-4.00
2025-03-05,UBER TRIP,-8.00
2025-03-08,MEDICAL PHARMACY,-25.00
2025-03-10,NETFLIX SUBSCRIPTION,-15.99
2025-03-12,FLIPKART ONLINE,-30.00
2025-03-15,DMART STORE,-45.00
2025-03-18,SALARY - ACME CORP,2500.00
2025-03-20,RECEIVED TRANSFER,100.00
2025-03-22,GROCERY MART,-55.00
2025-03-26,RECEIVED TRANSFER,200.00
2025-03-28,UBER TRIP,-6.50
2025-03-30,AMAZON MARKETPLACE,-70.00
2025-04-02,STARBUCKS COFFEE,-4.50
2025-04-05,SALARY - ACME CORP,2500.00
2025-04-07,GROCERY MART,-60.00
2025-04-10,FLIPKART ONLINE,-35.00
2025-04-12,UBER TRIP,-10.00
2025-04-15,DMART STORE,-40.00
2025-04-18,RECEIVED TRANSFER,150.00
2025-04-20,PHARMACY SHOP,-25.00
2025-04-25,AMAZON MARKETPLACE,-50.00
2025-04-28,STARBUCKS COFFEE,-5.00
2025-05-01,SALARY - ACME CORP,2500.00
2025-05-03,UBER TRIP,-12.00
2025-05-05,GROCERY MART,-55.00
2025-05-08,MEDICAL PHARMACY,-30.00
2025-05-10,FLIPKART ONLINE,-40.00
2025-05-12,RECEIVED TRANSFER,200.00
2025-05-15,DMART STORE,-45.00
2025-05-18,AMAZON MARKETPLACE,-60.00
2025-05-20,UBER TRIP,-8.50
2025-05-25,PHARMACY SHOP,-20.00
2025-05-28,STARBUCKS COFFEE,-4.50
"""

# -------------------------
# Sidebar: Upload + Sample Download
# -------------------------
with st.sidebar:
    st.header("Upload Statement")
    uploaded = st.file_uploader("CSV or Excel file", type=['csv','xlsx','xls'])
    st.markdown("---")
    st.markdown("Need a test file?")
    st.download_button(
        label="Download Sample CSV",
        data=SAMPLE_CSV,
        file_name="sample_statement.csv",
        mime="text/csv"
    )
    st.markdown("---")
    st.header("Quick Settings")
    show_raw_on_upload = st.checkbox("Show raw uploaded preview (sidebar)", value=False)
    st.caption("Use Settings tab to add custom category mappings.")

# If no upload, provide info and let user continue with sample
if uploaded is None:
    st.sidebar.info("No file uploaded — you can download the sample CSV above and then upload it to test the app.")

# -------------------------
# Main App Tabs
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 Trends", "🧾 Transactions", "⚙️ Settings"])

# Utility: load file if present
def load_dataframe(uploaded_file):
    try:
        if uploaded_file is None:
            return None
        name = uploaded_file.name.lower()
        if name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
        return df_raw
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None

df_raw = load_dataframe(uploaded)

if show_raw_on_upload and df_raw is not None:
    with st.sidebar.expander("Uploaded Raw Preview"):
        st.dataframe(df_raw.head(200))

# Settings: custom mapping input area (persist in session_state only)
if 'custom_map' not in st.session_state:
    st.session_state.custom_map = None

with tab4:
    st.header("⚙️ Settings")
    st.markdown("Add custom keyword mappings (JSON-like). Example: `{\"Dining\": [\"restaurant\",\"cafe\"]}`")
    custom_map_text = st.text_area("Custom mapping (optional)", value="" if st.session_state.custom_map is None else json.dumps(st.session_state.custom_map, indent=2), height=140)
    apply_map = st.button("Apply custom mapping")
    if apply_map:
        if custom_map_text.strip() == "":
            st.session_state.custom_map = None
            st.success("Cleared custom mapping.")
        else:
            try:
                parsed = json.loads(custom_map_text)
                if isinstance(parsed, dict):
                    st.session_state.custom_map = parsed
                    st.success("Custom mapping applied.")
                else:
                    st.error("Please provide a JSON object mapping category -> [keywords].")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

# If no file uploaded, still allow using sample as "uploaded"
if df_raw is None:
    # create df_raw from sample for previewing UI, but mark it as sample
    df_raw = pd.read_csv(StringIO(SAMPLE_CSV))
    is_sample = True
else:
    is_sample = False

# Try cleaning and categorizing; show errors gracefully
df_clean = None
try:
    df_clean = clean_and_normalize(df_raw)
    df_clean = categorize_transactions(df_clean, st.session_state.custom_map)
except Exception as e:
    # show a friendly message in each tab where content would be
    error_msg = str(e)
    # We'll show it below in each tab area
    df_clean = None

# ---------- TAB: Dashboard ----------
with tab1:
    st.header("📊 Dashboard")
    if df_clean is None:
        st.warning("No usable data available: " + error_msg if 'error_msg' in locals() else "Upload a CSV / Excel with Date, Description, Amount columns.")
    else:
        # KPIs
        total_income = df_clean.loc[df_clean['Amount'] > 0, 'Amount'].sum()
        total_expense = -df_clean.loc[df_clean['Amount'] < 0, 'Amount'].sum()
        net = total_income - total_expense

        k1, k2, k3, k4 = st.columns([1.3,1.3,1.3,1.3])
        with k1:
            st.markdown("<div class='card'><h4 style='margin:4px'>Total Income</h4><h2 style='color:{0}; margin:2px'>₹ {1:,.2f}</h2></div>".format(GREEN, total_income), unsafe_allow_html=True)
        with k2:
            st.markdown("<div class='card'><h4 style='margin:4px'>Total Expense</h4><h2 style='color:{0}; margin:2px'>₹ {1:,.2f}</h2></div>".format("#d9534f", total_expense), unsafe_allow_html=True)
        with k3:
            st.markdown("<div class='card'><h4 style='margin:4px'>Net</h4><h2 style='color:{0}; margin:2px'>₹ {1:,.2f}</h2></div>".format(ACCENT, net), unsafe_allow_html=True)
        with k4:
            avg_monthly = df_clean.groupby('YearMonth').agg(Net=('Amount','sum')).reset_index()
            avg_val = avg_monthly['Net'].mean() if len(avg_monthly)>0 else 0
            st.markdown("<div class='card'><h4 style='margin:4px'>Avg Monthly Net</h4><h2 style='color:{0}; margin:2px'>₹ {1:,.2f}</h2></div>".format(GREEN, avg_val), unsafe_allow_html=True)

        st.markdown("---")

        # Quick top categories
        st.subheader("Top Spending Categories")
        spend_by_cat = df_clean.loc[df_clean['Amount']<0].groupby('Category').agg(Amount=('Amount', lambda x: -x.sum())).sort_values('Amount', ascending=False).reset_index()
        if spend_by_cat.shape[0] == 0:
            st.info("No expense rows detected (negative amounts).")
        else:
            top_cats = spend_by_cat.head(5)
            cols = st.columns(len(top_cats))
            for c_idx, row in enumerate(top_cats.itertuples()):
                with cols[c_idx]:
                    st.metric(label=row.Category, value=f"₹ {row.Amount:,.2f}")

        st.markdown("---")
        st.subheader("Recent Transactions")
        recent = df_clean.sort_values('Date', ascending=False).head(8)
        st.table(recent[['Date','Description','Amount','Category']].assign(Amount=lambda d: d['Amount'].map(lambda x: f"₹ {x:,.2f}")))

# ---------- TAB: Trends ----------
with tab2:
    st.header("📈 Trends")
    if df_clean is None:
        st.warning("No usable data available: " + error_msg if 'error_msg' in locals() else "Upload a CSV / Excel with Date, Description, Amount columns.")
    else:
        # Monthly aggregates
        monthly = df_clean.groupby('YearMonth').agg(
            Income = ('Amount', lambda x: x[x>0].sum()),
            Expense = ('Amount', lambda x: -x[x<0].sum()),
            Net = ('Amount', 'sum')
        ).reset_index()
        monthly['YearMonth_dt'] = pd.to_datetime(monthly['YearMonth'] + "-01")
        monthly = monthly.sort_values('YearMonth_dt')

        # Line chart (seaborn)
        fig, ax = plt.subplots(figsize=(10,4))
        sns.lineplot(x='YearMonth_dt', y='Income', data=monthly, label='Income', marker='o')
        sns.lineplot(x='YearMonth_dt', y='Expense', data=monthly, label='Expense', marker='o')
        sns.lineplot(x='YearMonth_dt', y='Net', data=monthly, label='Net', marker='o')
        ax.set_title("Monthly Income / Expense / Net")
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig, use_container_width=True)

        st.markdown("---")
        # Savings predictor (simple linear fit)
        st.subheader("Savings predictor (next month)")
        savings_series = monthly[['YearMonth_dt','Net']].copy().reset_index(drop=True)
        savings_series['x'] = np.arange(len(savings_series))
        if len(savings_series) >= 3:
            coeffs = np.polyfit(savings_series['x'], savings_series['Net'], 1)
            slope, intercept = coeffs[0], coeffs[1]
            next_x = savings_series['x'].max() + 1
            next_pred = slope * next_x + intercept
            st.write(f"Predicted next month's net (income - expense): **₹ {next_pred:,.2f}** (linear trend)")
            fig2, ax2 = plt.subplots(figsize=(9,3))
            ax2.plot(savings_series['YearMonth_dt'], savings_series['Net'], marker='o', label='Net')
            future_dt = savings_series['YearMonth_dt'].max() + pd.offsets.MonthBegin(1)
            ax2.plot(list(savings_series['YearMonth_dt']) + [future_dt],
                     np.append(np.polyval(coeffs, savings_series['x']), next_pred),
                     linestyle='--', marker='x', label='Trend & next prediction')
            ax2.set_title("Net Savings and Linear Trend")
            ax2.tick_params(axis='x', rotation=45)
            ax2.legend()
            st.pyplot(fig2, use_container_width=True)
        else:
            st.info("Need at least 3 months of data for a trend-based prediction.")

# ---------- TAB: Transactions ----------
with tab3:
    st.header("🧾 Transactions")
    if df_clean is None:
        st.warning("No usable data available: " + error_msg if 'error_msg' in locals() else "Upload a CSV / Excel with Date, Description, Amount columns.")
    else:
        filt_col1, filt_col2, filt_col3 = st.columns([1,1,1])
        with filt_col1:
            cat_filter = st.selectbox("Category", options=["All"] + sorted(df_clean['Category'].unique().tolist()))
        with filt_col2:
            from_date = st.date_input("From", value=df_clean['Date'].min().date())
        with filt_col3:
            to_date = st.date_input("To", value=df_clean['Date'].max().date())

        mask = (df_clean['Date'].dt.date >= from_date) & (df_clean['Date'].dt.date <= to_date)
        if cat_filter != "All":
            mask = mask & (df_clean['Category'] == cat_filter)
        filtered = df_clean.loc[mask].sort_values('Date', ascending=False)

        st.markdown(f"Showing **{len(filtered)}** transactions")
        st.dataframe(filtered[['Date','Description','Amount','Category']].assign(Amount=lambda d: d['Amount'].map(lambda x: f"₹ {x:,.2f}")))

        # Download filtered CSV
        csv_bytes = filtered.to_csv(index=False).encode('utf-8')
        st.download_button("Download filtered CSV", data=csv_bytes, file_name="transactions_filtered.csv", mime="text/csv")

        st.markdown("---")
        # Top Expenses
        st.subheader("Top Expenses")
        top_expenses = df_clean.loc[df_clean['Amount']<0].sort_values('Amount').head(10)
        if top_expenses.shape[0] == 0:
            st.info("No expenses found.")
        else:
            st.table(top_expenses[['Date','Description','Amount','Category']].assign(Amount=lambda d: d['Amount'].map(lambda x: f"₹ {-x:,.2f}")))

# Final footer / note
st.markdown("---")
st.caption("Built with Pandas · NumPy · Matplotlib · Seaborn · Streamlit — Theme: Finance Green 🍃")
