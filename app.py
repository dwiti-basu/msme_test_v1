import os
import streamlit as st
import pandas as pd
import numpy as np
from src import utils
from packs import retail, services, saas

st.set_page_config(page_title="MSME Analytics – Pro", layout="wide")
st.title("📊 MSME Analytics – Pro (Multi-vertical)")

def data_readiness(df: pd.DataFrame, required_cols, nice_name="Dataset"):
    st.subheader("Data readiness")
    missing = [c for c in required_cols if c not in df.columns]
    cols = st.columns([3, 7])
    cols[0].metric(f"{nice_name} rows", f"{len(df):,}")
    cols[1].write("**Columns present:** " + ", ".join(df.columns.astype(str).tolist()))
    if not missing:
        st.success("All required columns found ✔️")
        return True
    else:
        st.error("Missing required columns: " + ", ".join(missing))
        return False

pack_name = st.selectbox("Business type", [
    "Retail / D2C","Services / Agency","Subscription / SaaS"
], index=0)

use_demo = st.toggle("Use demo dataset", value=True)
mask_demo = st.checkbox("Mask values (demo safe)", value=False)

if pack_name == "Retail / D2C":
    if use_demo:
        path = os.path.join("samples","retail_orders.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
        else:
            st.warning("Demo file not found. Upload your file below.")
            up = st.file_uploader("Upload retail orders (CSV/XLSX)", type=["csv","xlsx","xls"])
            if not up: st.stop()
            df = utils.normalize_retail(up.getvalue(), up.name)
    else:
        up = st.file_uploader("Upload retail orders (CSV/XLSX)", type=["csv","xlsx","xls"])
        if not up: st.stop()
        df = utils.normalize_retail(up.getvalue(), up.name)

    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce').dt.date
        df = df.dropna(subset=['order_date'])

    if mask_demo and 'revenue' in df:
        df['revenue'] = df['revenue'] * np.random.uniform(0.7, 1.3)

    with st.expander("Preview", expanded=False):
        st.dataframe(df.head(50), use_container_width=True)

    if not data_readiness(df, ["order_date","sku","revenue"], "Retail"):
        st.stop()

    try:
        retail.Analyzer().run(df)
    except Exception as e:
        st.exception(e)

elif pack_name == "Services / Agency":
    if use_demo:
        path = os.path.join("samples","services_timesheets.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=['invoice_date','paid_date'])
        else:
            st.warning("Demo file not found. Upload your file below.")
            up = st.file_uploader("Upload services timesheets (CSV/XLSX)", type=["csv","xlsx","xls"])
            if not up: st.stop()
            df = utils.normalize_services(up.getvalue(), up.name)
    else:
        up = st.file_uploader("Upload services timesheets (CSV/XLSX)", type=["csv","xlsx","xls"])
        if not up: st.stop()
        df = utils.normalize_services(up.getvalue(), up.name)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        df = df.dropna(subset=['date'])

    if 'revenue' not in df.columns:
        if 'invoice_amount' in df.columns:
            df['revenue'] = df['invoice_amount']
        elif {'hours','rate','billable'}.issubset(df.columns):
            df['revenue'] = df['hours'] * df['rate'] * df['billable'].astype(int)
        else:
            df['revenue'] = 0.0

    with st.expander("Preview", expanded=False):
        st.dataframe(df.head(50), use_container_width=True)

    if not data_readiness(df, ["date","revenue"], "Services"):
        st.stop()

    try:
        services.Analyzer().run(df)
    except Exception as e:
        st.exception(e)

else:
    if use_demo:
        path = os.path.join("samples","saas_subscriptions.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=['start','end','event_ts'])
        else:
            st.warning("Demo file not found. Upload your file below.")
            up = st.file_uploader("Upload SaaS subscriptions/events (CSV/XLSX)", type=["csv","xlsx","xls"])
            if not up: st.stop()
            df = utils.normalize_saas(up.getvalue(), up.name)
    else:
        up = st.file_uploader("Upload SaaS subscriptions/events (CSV/XLSX)", type=["csv","xlsx","xls"])
        if not up: st.stop()
        df = utils.normalize_saas(up.getvalue(), up.name)

    for c in ['start','end','event_ts']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')

    with st.expander("Preview", expanded=False):
        st.dataframe(df.head(50), use_container_width=True)

    if not data_readiness(df, ["customer_id","plan","start"], "SaaS"):
        st.stop()

    try:
        saas.Analyzer().run(df)
    except Exception as e:
        st.exception(e)
