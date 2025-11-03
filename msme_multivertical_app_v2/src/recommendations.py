import pandas as pd
import numpy as np
from .utils import (
    wow_growth, abc_pareto, anomalies_zscore,
    cohort_retention, saas_mrr_movements
)

def retail_recos(df: pd.DataFrame):
    recos = []
    total_rev = float(df['revenue'].sum())
    gm_pct = None
    if 'cost' in df:
        gm = float((df['revenue'] - df['cost'].fillna(0)).sum())
        gm_pct = (gm / total_rev * 100) if total_rev else None
    daily = df.groupby('order_date', as_index=False)['revenue'].sum().sort_values('order_date')
    gow = wow_growth(daily)
    pareto = abc_pareto(df)
    top_sku_share = float(pareto['revenue'].iloc[0] / pareto['revenue'].sum()) if not pareto.empty and pareto['revenue'].sum() else 0.0
    if 'location' in df:
        loc = df.groupby('location', as_index=False)['revenue'].sum().sort_values('revenue', ascending=False)
        top_loc_share = float(loc['revenue'].iloc[0] / loc['revenue'].sum()) if not loc.empty else 0.0
    else:
        top_loc_share = 0.0
    an = anomalies_zscore(daily)
    neg_anoms = an[an['z'] < -2] if not an.empty else pd.DataFrame()
    discount_margin_flag = False
    if {'discount','price','cost'}.issubset(df.columns):
        d = df.copy()
        d['disc_pct'] = np.where(d['price'].notna() & (d['price']>0), d['discount']/d['price']*100, 0)
        d['margin_pct'] = np.where(d['revenue']>0, (d['revenue']-d['cost'].fillna(0))/d['revenue']*100, np.nan)
        dm = d[['disc_pct','margin_pct']].dropna()
        if dm.shape[0] >= 10:
            corr = dm.corr().iloc[0,1]
            discount_margin_flag = (pd.notna(corr) and corr < -0.2)
    m1_ret = None
    try:
        ret = cohort_retention(df)
        if ret.shape[1] > 1:
            m1_col = 1 if 1 in ret.columns else ret.columns.min()
            m1_ret = float(ret[m1_col].mean())
    except Exception:
        pass
    if gow is not None and gow < 0:
        recos.append(f"WoW revenue down {gow:.1f}% — audit ads/inventory/logistics on last 7 days; recover top {min(3, len(neg_anoms))} negative anomalies first.")
    if top_sku_share >= 0.40:
        recos.append("Revenue concentration risk: top SKU ≥40% of sales — secure supply, add 1–2 adjacent variants, and diversify placements.")
    if top_loc_share >= 0.40:
        recos.append("Geographic concentration risk: top location ≥40% — pilot 1–2 new cities/marketplaces with best-fit SKUs.")
    if discount_margin_flag:
        recos.append("Discounts appear to erode margin — cap promo depth on low-elastic SKUs; shift to bundles/loyalty points.")
    if gm_pct is not None and gm_pct < 35:
        recos.append("Gross margin <35% — renegotiate vendor terms/MOQs, optimize freight, and prune negative-margin SKUs.")
    if m1_ret is not None and m1_ret < 0.25:
        recos.append("Weak cohort month-1 retention — add day-7/day-30 WhatsApp nudges, refill reminders, and how-to content.")
    recos.append("Deploy ‘Frequently Bought Together’ from Basket Pairs to lift AOV by 5–8%.")
    return recos

def services_recos(df: pd.DataFrame):
    recos = []
    total_hours = float(df['hours'].sum()) if 'hours' in df else 0.0
    billable_hours = float(df.loc[df.get('billable', True)==True, 'hours'].sum()) if 'hours' in df else 0.0
    util = (billable_hours / total_hours * 100) if total_hours > 0 else None
    revenue = float(df.get('revenue', pd.Series([0])).sum())
    denom = float((df['hours']*df['rate']).sum()) if {'hours','rate'}.issubset(df.columns) else 0.0
    realization = (revenue / denom * 100) if denom > 0 else None
    top_client_share = 0.0
    if 'client' in df:
        rc = df.groupby('client', as_index=False)['revenue'].sum().sort_values('revenue', ascending=False)
        top_client_share = float(rc['revenue'].iloc[0] / rc['revenue'].sum()) if not rc.empty else 0.0
    ar_91p_share = None
    if {'invoice_date','invoice_amount'}.issubset(df.columns):
        inv = df[['invoice_date','invoice_amount','paid_date']].copy()
        inv['invoice_date'] = pd.to_datetime(inv['invoice_date'], errors='coerce')
        inv['paid_date'] = pd.to_datetime(inv.get('paid_date', pd.NaT), errors='coerce')
        today = inv['invoice_date'].max() + pd.Timedelta(days=1)
        inv['open_amt'] = np.where(inv['paid_date'].notna(), 0, inv['invoice_amount'])
        inv['age'] = (today - inv['invoice_date']).dt.days
        ar_tot = inv['open_amt'].sum()
        ar_91p = inv.loc[inv['age'] >= 91, 'open_amt'].sum()
        ar_91p_share = (ar_91p / ar_tot) if ar_tot > 0 else None
    if util is not None and util < 70:
        recos.append("Under-utilization <70% — rebalance bench to active projects and tighten weekly staffing reviews.")
    if realization is not None and realization < 85:
        recos.append("Realization <85% — enforce change-orders for scope creep and review discounting on low-margin tasks.")
    if top_client_share >= 0.35:
        recos.append("Client concentration risk (>35%) — add 1–2 smaller retainers to dilute exposure.")
    if ar_91p_share is not None and ar_91p_share > 0.25:
        recos.append("Collections risk: >25% A/R is 91+ days — activate dunning cadence and early-pay incentives (2/10, net-30).")
    recos.append("Build a 12-week capacity plan targeting 75–80% utilization; hire only if forecast >85% for 3 consecutive weeks.")
    return recos

def saas_recos(df: pd.DataFrame):
    recos = []
    subs = df[['customer_id','plan','start','end','mrr','cac']].copy()
    subs['start'] = pd.to_datetime(subs['start'], errors='coerce')
    subs['end'] = pd.to_datetime(subs['end'], errors='coerce')
    mv = saas_mrr_movements(subs)
    if mv.empty:
        return ["Add at least two months of subscription data to enable MRR movement recommendations."]
    mrr_now = float(mv['MRR'].iloc[-1])
    net_last = float(mv['Net'].iloc[-1])
    churn_last = float(mv['Churn'].iloc[-1])
    contraction_last = float(mv['Contraction'].iloc[-1])
    expansion_last = float(mv['Expansion'].iloc[-1])
    plan_mix_flag = False
    if 'plan' in df:
        mix = df.groupby('plan')['customer_id'].nunique().sort_values(ascending=False)
        total_cust = mix.sum()
        if total_cust > 0:
            top_plan_share = float(mix.iloc[0] / total_cust)
            plan_mix_flag = top_plan_share > 0.60
    payback_mo = None
    if 'cac' in df.columns and df['cac'].notna().any():
        cur = subs[(subs['end'].isna()) | (subs['end'] >= subs['start'].max())]
        customers = cur['customer_id'].nunique()
        arpu = mrr_now / max(customers, 1)
        gross_margin = 0.8
        avg_cac = float(df['cac'].dropna().mean())
        if arpu * gross_margin > 0:
            payback_mo = avg_cac / (arpu * gross_margin)
    if churn_last > 0 and (expansion_last - contraction_last) < churn_last:
        recos.append("Churn outpaces expansion — add save-offers (pause plan), and drive activation on premium features.")
    if net_last <= 0:
        recos.append("Net MRR non-positive — tighten onboarding to day-1 ‘Aha!’ and focus PMM on upgrade paths.")
    if plan_mix_flag:
        recos.append("Plan mix skewed to entry tier — introduce an in-between plan with clear upgrade benefits; test seat-based pricing.")
    if payback_mo is not None and payback_mo > 12:
        recos.append(f"CAC payback ≈{payback_mo:.1f} months (>12) — shift spend to higher-conversion channels or lift ARPU via packaging.")
    recos.append("Track NDR monthly; target ≥110% (SMB) / ≥120% (mid-market).")
    return recos
