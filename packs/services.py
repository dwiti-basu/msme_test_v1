import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils import ar_aging, utilization_heatmap
from src.recommendations import services_recos

class Analyzer:
    name = "Services / Agency"
    def run(self, df: pd.DataFrame):
        if 'revenue' not in df.columns:
            if 'invoice_amount' in df.columns: df['revenue'] = df['invoice_amount']
            elif {'hours','rate','billable'}.issubset(df.columns): df['revenue'] = df['hours']*df['rate']*df['billable'].astype(int)
            else: df['revenue'] = 0.0

        total_hours = df['hours'].sum() if 'hours' in df else 0
        billable_hours = df.loc[df.get('billable', True)==True, 'hours'].sum() if 'hours' in df else 0
        utilization = float(billable_hours / total_hours * 100) if total_hours>0 else None
        revenue = df['revenue'].sum()
        denom = (df['hours']*df['rate']).sum() if {'hours','rate'}.issubset(df.columns) else 0
        realization = float(revenue / denom * 100) if denom>0 else None

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Revenue", f"₹{revenue:.0f}")
        c2.metric("Total Hours", f"{total_hours:.1f}")
        c3.metric("Utilization %", f"{utilization:.1f}%" if utilization is not None else "—")
        c4.metric("Realization %", f"{realization:.1f}%" if realization is not None else "—")

        tabs = st.tabs(["Utilization by Consultant","Revenue by Client","Project Margin %","Utilization Heatmap","A/R Aging","Forecast (This Month)"])

        with tabs[0]:
            if 'hours' in df and 'consultant' in df:
                util = df.groupby('consultant').agg(hours=('hours','sum'),
                                                    billable_hours=('billable', lambda s: (s==True).sum() if s.notna().any() else 0)).reset_index()
                util['util_pct'] = np.where(util['hours']>0, util['billable_hours']/util['hours']*100, 0)
                st.dataframe(util, use_container_width=True)
                fig, ax = plt.subplots(); ax.bar(util['consultant'], util['util_pct']); ax.set_title("Utilization % by Consultant"); ax.set_xlabel("Consultant"); ax.set_ylabel("%"); st.pyplot(fig)
            else:
                st.info("Add 'hours' and 'consultant' columns to see utilization.")

        with tabs[1]:
            if 'client' in df:
                rc = df.groupby('client').agg(revenue=('revenue','sum')).reset_index()
                st.dataframe(rc.sort_values('revenue', ascending=False), use_container_width=True)
                fig2, ax2 = plt.subplots(); ax2.bar(rc['client'], rc['revenue']); ax2.set_title("Revenue by Client"); ax2.set_xlabel("Client"); ax2.set_ylabel("Revenue"); st.pyplot(fig2)
            else:
                st.info("Add 'client' to see revenue by client.")

        with tabs[2]:
            if {'hours','rate','project'}.issubset(df.columns):
                df['cost'] = df['hours'] * (df['rate']*0.6)  # proxy
                proj = df.groupby('project').agg(revenue=('revenue','sum'), cost=('cost','sum')).reset_index()
                proj['margin_pct'] = np.where(proj['revenue']>0, (proj['revenue']-proj['cost'])/proj['revenue']*100, 0)
                st.dataframe(proj.sort_values('margin_pct'), use_container_width=True)
                fig3, ax3 = plt.subplots(); ax3.bar(proj['project'], proj['margin_pct']); ax3.set_title("Project Margin %"); ax3.set_xlabel("Project"); ax3.set_ylabel("Margin %"); st.pyplot(fig3)
            else:
                st.info("Need 'hours','rate','project' to compute project margin.")

        with tabs[3]:
            if {'date','consultant','hours'}.issubset(df.columns):
                piv = utilization_heatmap(df)
                st.dataframe(piv, use_container_width=True)
                fig4, ax4 = plt.subplots(); ax4.imshow(piv.values, aspect='auto'); ax4.set_title("Utilization Heatmap (Hours)"); ax4.set_xlabel("Week #"); ax4.set_ylabel("Consultant"); st.pyplot(fig4)
            else:
                st.info("Need 'date','consultant','hours' to draw heatmap.")

        with tabs[4]:
            if {'invoice_date','invoice_amount'}.issubset(df.columns):
                aging = ar_aging(df[['invoice_date','invoice_amount','paid_date']])
                st.dataframe(aging, use_container_width=True)
                fig5, ax5 = plt.subplots(); ax5.bar(aging['bucket'], aging['Amount']); ax5.set_title("A/R Aging"); ax5.set_xlabel("Bucket"); ax5.set_ylabel("Open Amount"); st.pyplot(fig5)
            else:
                st.info("Add 'invoice_date' and 'invoice_amount' to compute A/R aging.")

        with tabs[5]:
            if 'date' in df:
                d = df.copy(); d['date'] = pd.to_datetime(d['date'], errors='coerce')
                month_df = d[d['date'].dt.to_period('M') == d['date'].max().to_period('M')]
                days_elapsed = month_df['date'].nunique()
                month_total_so_far = month_df['revenue'].sum()
                forecast = (month_total_so_far / max(days_elapsed,1)) * 30
                st.metric("Forecast Revenue (Month)", f"₹{forecast:,.0f}")
            else:
                st.info("Need 'date' to compute a simple monthly run-rate forecast.")

        st.subheader("🧭 Recommendations")
        for tip in services_recos(df):
            st.markdown(f"- {tip}")
