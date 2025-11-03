import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils import wow_growth, abc_pareto, anomalies_zscore, rfm, cohort_retention, price_volume_mix, yoy_monthly, basket_pairs
from src.recommendations import retail_recos

class Analyzer:
    name = "Retail / D2C"
    def run(self, df: pd.DataFrame):
        total_rev = df['revenue'].sum()
        gm = (df['revenue'] - df['cost'].fillna(0)).sum() if 'cost' in df else np.nan
        gm_pct = float(gm/total_rev*100) if total_rev and pd.notna(gm) else None
        orders = len(df)
        aov = float(total_rev/orders) if orders else None
        discounts = df['discount'].sum() if 'discount' in df else 0
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Revenue", f"₹{total_rev:.0f}")
        c2.metric("GM%", f"{gm_pct:.1f}%" if gm_pct is not None else "—")
        c3.metric("Orders", f"{orders}")
        c4.metric("AOV", f"₹{aov:.0f}" if aov else "—")
        if isinstance(discounts,(int,float)): st.caption(f"Discounts booked: ₹{discounts:,.0f}")

        daily = df.groupby('order_date', as_index=False)['revenue'].sum().sort_values('order_date')
        g = wow_growth(daily)
        if g is not None:
            st.info(f"Auto-insight: WoW growth {'↑' if g>=0 else '↓'} {g:.1f}%")

        st.subheader("Daily Revenue")
        fig, ax = plt.subplots()
        ax.plot(pd.to_datetime(daily['order_date']), daily['revenue'])
        ax.set_xlabel("Date"); ax.set_ylabel("Revenue"); ax.set_title("Daily Revenue")
        st.pyplot(fig)

        tabs = st.tabs(["Pareto & ABC","Anomalies","SKU & Location","RFM","Cohorts","Price–Volume Mix","YoY by Month","Basket Pairs","Discount vs Margin","Channel Mix"])

        with tabs[0]:
            sk = abc_pareto(df); st.dataframe(sk[['sku','revenue','cum_share','ABC']].head(50), use_container_width=True)
            fig2, ax2 = plt.subplots()
            if not sk.empty: ax2.plot(range(1, len(sk['cum_share'])+1), sk['cum_share'])
            ax2.set_xlabel("SKU rank"); ax2.set_ylabel("Cum. revenue share"); ax2.set_title("Pareto curve")
            st.pyplot(fig2)

        with tabs[1]:
            an = anomalies_zscore(daily); st.dataframe(an, use_container_width=True)
            fig3, ax3 = plt.subplots(); ax3.plot(pd.to_datetime(daily['order_date']), daily['revenue'])
            if len(an)>0: ax3.scatter(pd.to_datetime(an['order_date']), an['revenue'])
            ax3.set_title("Daily Revenue with anomalies")
            st.pyplot(fig3)

        with tabs[2]:
            top_skus = df.groupby('sku', as_index=False)['revenue'].sum().sort_values('revenue', ascending=False).head(20)
            st.write("Top SKUs"); st.dataframe(top_skus, use_container_width=True)
            fig4, ax4 = plt.subplots(); ax4.bar(top_skus['sku'], top_skus['revenue']); ax4.set_title("Top SKUs"); ax4.set_xlabel("SKU"); ax4.set_ylabel("Revenue"); st.pyplot(fig4)
            top_loc = df.groupby('location', as_index=False)['revenue'].sum().sort_values('revenue', ascending=False).head(20)
            st.write("Top Locations"); st.dataframe(top_loc, use_container_width=True)
            fig5, ax5 = plt.subplots(); ax5.bar(top_loc['location'], top_loc['revenue']); ax5.set_title("Top Locations"); ax5.set_xlabel("Location"); ax5.set_ylabel("Revenue"); st.pyplot(fig5)

        with tabs[3]:
            if 'customer_id' in df and df['customer_id'].nunique() > 0:
                r = rfm(df); st.dataframe(r[['customer_id','recency','freq','monetary','R','F','M','Segment']].sort_values('monetary', ascending=False).head(100), use_container_width=True)
                seg_counts = r['Segment'].value_counts().sort_values(ascending=False)
                fig6, ax6 = plt.subplots(); ax6.bar(seg_counts.index.astype(str), seg_counts.values); ax6.set_title("Customers by RFM Segment"); ax6.set_xlabel("Segment"); ax6.set_ylabel("Count"); st.pyplot(fig6)
            else:
                st.info("Customer IDs missing; RFM/Cohorts will be limited.")

        with tabs[4]:
            try:
                ret = cohort_retention(df); st.dataframe(ret, use_container_width=True)
                if ret.shape[0] > 0 and ret.shape[1] > 0:
                    fig7, ax7 = plt.subplots(); ax7.imshow(ret.values, aspect='auto'); ax7.set_title("Retention heatmap"); ax7.set_xlabel("Months since"); ax7.set_ylabel("Cohort"); st.pyplot(fig7)
            except Exception as e:
                st.info("Cohort view unavailable for this dataset.")
                st.exception(e)

        with tabs[5]:
            pvm = price_volume_mix(df); st.dataframe(pvm, use_container_width=True)
            if not pvm.empty:
                fig8, ax8 = plt.subplots(); ax8.bar(['Price Effect','Volume Effect','Total Change'], [pvm.iloc[-1]['price_effect'], pvm.iloc[-1]['volume_effect'], pvm.iloc[-1]['total_change']]); ax8.set_title("Latest Month Change"); st.pyplot(fig8)

        with tabs[6]:
            yoy = yoy_monthly(df, 'order_date', 'revenue'); st.dataframe(yoy, use_container_width=True)
            if yoy.shape[1] >= 2:
                fig9, ax9 = plt.subplots()
                for col in yoy.columns:
                    ax9.plot(yoy.index, yoy[col], label=str(col))
                ax9.set_title("YoY Monthly Revenue"); ax9.set_xlabel("Month (1-12)"); ax9.set_ylabel("Revenue"); st.pyplot(fig9)

        with tabs[7]:
            pairs = basket_pairs(df); st.dataframe(pairs, use_container_width=True)
            if not pairs.empty:
                fig10, ax10 = plt.subplots(); ax10.barh((pairs['item_a']+" + "+pairs['item_b']).iloc[::-1], pairs['count'].iloc[::-1]); ax10.set_title("Top Basket Pairs"); st.pyplot(fig10)

        with tabs[8]:
            if 'discount' in df.columns and 'price' in df.columns and 'cost' in df.columns:
                d = df.copy()
                d['margin_pct'] = np.where(d['revenue']>0, (d['revenue']-d['cost'].fillna(0))/d['revenue']*100, np.nan)
                d['disc_pct'] = np.where(d['price'].notna() & (d['price']>0), d['discount']/d['price']*100, 0)
                fig11, ax11 = plt.subplots(); ax11.scatter(d['disc_pct'], d['margin_pct']); ax11.set_xlabel("Discount %"); ax11.set_ylabel("GM %"); ax11.set_title("Discount vs Margin"); st.pyplot(fig11)
            else:
                st.info("Need 'discount', 'price' and 'cost' columns to analyze discount vs margin.")

        with tabs[9]:
            if 'channel' in df:
                ch = df.groupby('channel', as_index=False)['revenue'].sum().sort_values('revenue', ascending=False)
                st.dataframe(ch, use_container_width=True)
                fig12, ax12 = plt.subplots(); ax12.bar(ch['channel'], ch['revenue']); ax12.set_title("Channel Mix"); ax12.set_xlabel("Channel"); ax12.set_ylabel("Revenue"); st.pyplot(fig12)
            else:
                st.info("No 'channel' column found.")

        st.subheader("🧭 Recommendations")
        for tip in retail_recos(df):
            st.markdown(f"- {tip}")
