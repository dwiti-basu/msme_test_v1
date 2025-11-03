import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils import saas_mrr_movements
from src.recommendations import saas_recos

class Analyzer:
    name = "Subscription / SaaS"
    def run(self, df: pd.DataFrame):
        if df.empty:
            st.warning("No data loaded.")
            return
        subs = df[['customer_id','plan','start','end','mrr','cac']].copy()
        subs['start'] = pd.to_datetime(subs['start'], errors='coerce')
        subs['end'] = pd.to_datetime(subs['end'], errors='coerce')

        mv = saas_mrr_movements(subs)
        st.subheader("Net MRR Movements")
        st.dataframe(mv, use_container_width=True)

        c1,c2,c3,c4,c5 = st.columns(5)
        mrr_now = float(mv['MRR'].iloc[-1]) if not mv.empty else 0.0
        arr = mrr_now*12
        net = float(mv['Net'].iloc[-1]) if not mv.empty else 0.0
        growth = float(((mv['MRR'].iloc[-1]/mv['MRR'].iloc[-2])-1)*100) if len(mv)>=2 and mv['MRR'].iloc[-2]>0 else None
        c1.metric("MRR (current)", f"₹{mrr_now:,.0f}")
        c2.metric("ARR", f"₹{arr:,.0f}")
        c3.metric("Net Movement (last month)", f"₹{net:,.0f}")
        c4.metric("MRR Growth % (MoM)", f"{growth:.1f}%" if growth is not None else "—")
        c5.metric("Months tracked", f"{len(mv)}")

        fig, ax = plt.subplots(); ax.plot(mv['month'], mv['MRR']); ax.set_title("MRR over time"); ax.set_xlabel("Month"); ax.set_ylabel("MRR"); st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        x = mv['month'].astype(str)
        ax2.bar(x, mv['New'], label='New')
        ax2.bar(x, mv['Expansion'], bottom=mv['New'], label='Expansion')
        ax2.bar(x, -mv['Contraction'], label='Contraction')
        ax2.bar(x, -mv['Churn'], bottom=-mv['Contraction'], label='Churn')
        ax2.set_title("MRR Movements"); ax2.set_xticklabels(x, rotation=45)
        st.pyplot(fig2)

        mix = df.groupby('plan')['customer_id'].nunique().reset_index(name='customers').sort_values('customers', ascending=False)
        st.subheader("Plan Mix")
        st.dataframe(mix, use_container_width=True)
        fig3, ax3 = plt.subplots(); ax3.bar(mix['plan'], mix['customers']); ax3.set_title("Customers by Plan"); ax3.set_xlabel("Plan"); ax3.set_ylabel("Customers"); st.pyplot(fig3)

        if 'cac' in subs.columns and subs['cac'].notna().any():
            cur = subs[(subs['end'].isna()) | (subs['end'] >= subs['start'].max())]
            customers = cur['customer_id'].nunique()
            arpu = mrr_now / max(customers,1)
            if len(mv)>=2 and mv['MRR'].iloc[-2]>0:
                churn_rate = mv['Churn'].iloc[-1] / mv['MRR'].iloc[-2]
            else:
                churn_rate = np.nan
            gross_margin = 0.8
            if pd.notna(churn_rate) and churn_rate>0:
                ltv = (arpu * gross_margin) / churn_rate
                avg_cac = subs['cac'].dropna().mean()
                st.metric("Estimated LTV", f"₹{ltv:,.0f}")
                st.metric("Avg CAC", f"₹{avg_cac:,.0f}")
                st.write(f"Payback months (approx): {avg_cac / max(arpu*gross_margin,1):.1f}")
            else:
                st.info("Not enough data to estimate churn-based LTV.")

        st.subheader("🧭 Recommendations")
        for tip in saas_recos(df):
            st.markdown(f"- {tip}")
