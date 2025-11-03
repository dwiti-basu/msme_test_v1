import pandas as pd
import numpy as np
from io import BytesIO
from typing import Dict

def infer_columns(df: pd.DataFrame, mapping: Dict[str, list]) -> Dict[str, str]:
    cols = {c.lower().strip(): c for c in df.columns}
    result = {}
    for canon, options in mapping.items():
        result[canon] = None
        for opt in options:
            for k in cols:
                if opt in k:
                    result[canon] = cols[k]
                    break
            if result[canon]:
                break
    return result

def normalize_retail(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith((".xlsx",".xls")):
        df = pd.read_excel(BytesIO(file_bytes))
    else:
        df = pd.read_csv(BytesIO(file_bytes))
    mapping = infer_columns(df, {
        "date": ["order_date","invoice_date","bill_date","date"],
        "order_id": ["order_id","invoice_no","order","order number","doc"],
        "sku": ["sku","item_code","product_code","item","asin","product id"],
        "product_name": ["product","item","name","description","title"],
        "location": ["city","region","location","store","branch","state"],
        "qty": ["qty","quantity","units","unit","pieces"],
        "price": ["price","rate","unit_price","selling price"],
        "revenue": ["revenue","sales","amount","net_sales","total_value","total"],
        "discount": ["discount","disc","promo","markdown"],
        "cost": ["cost","cogs","unit_cost","purchase_cost"],
        "customer_id": ["customer_id","client_id","party","customer","buyer_id","phone","email"],
        "channel": ["channel","source","marketing_channel","utm"],
    })
    def col(key, default=None):
        c = mapping.get(key)
        return df[c] if c and c in df.columns else pd.Series([default]*len(df))
    norm = pd.DataFrame({
        "order_date": pd.to_datetime(col("date"), errors='coerce').dt.date,
        "order_id": col("order_id").astype(str) if mapping.get("order_id") else pd.Series([None]*len(df)),
        "sku": col("sku","").astype(str),
        "product_name": col("product_name","").astype(str),
        "location": col("location","ALL").astype(str),
        "qty": pd.to_numeric(col("qty",1), errors="coerce").fillna(1),
        "price": pd.to_numeric(col("price", np.nan), errors="coerce"),
        "revenue": pd.to_numeric(col("revenue", np.nan), errors="coerce"),
        "discount": pd.to_numeric(col("discount", 0), errors="coerce").fillna(0),
        "cost": pd.to_numeric(col("cost", np.nan), errors="coerce"),
        "customer_id": col("customer_id","CUST_UNKNOWN").astype(str),
        "channel": col("channel","Unattributed").astype(str),
    })
    norm["revenue"] = norm["revenue"].fillna(norm["qty"]*norm["price"] - norm["discount"])
    if norm["cost"].isna().all() and norm["price"].notna().any():
        norm["cost"] = norm["qty"] * norm["price"] * 0.7
    return norm.dropna(subset=["order_date","sku","revenue"])

def normalize_services(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith((".xlsx",".xls")):
        df = pd.read_excel(BytesIO(file_bytes))
    else:
        df = pd.read_csv(BytesIO(file_bytes))
    mapping = infer_columns(df, {
        "date": ["date","work_date","entry_date","timesheet_date"],
        "consultant": ["consultant","employee","resource","staff","user"],
        "project": ["project","engagement","job","task"],
        "hours": ["hours","hrs","duration","time_spent"],
        "rate": ["rate","hourly_rate","billing_rate"],
        "billable": ["billable","is_billable"],
        "client": ["client","customer","account"],
        "invoice_amount": ["amount","invoice_amount","billed_amount","invoice_total"],
        "invoice_date": ["invoice_date","bill_date","billed_on"],
        "paid_date": ["paid_date","payment_date","received_on"],
    })
    def col(key, default=None):
        c = mapping.get(key)
        return df[c] if c and c in df.columns else pd.Series([default]*len(df))
    norm = pd.DataFrame({
        "date": pd.to_datetime(col("date"), errors='coerce').dt.date,
        "consultant": col("consultant","").astype(str),
        "project": col("project","").astype(str),
        "hours": pd.to_numeric(col("hours",0), errors="coerce").fillna(0),
        "rate": pd.to_numeric(col("rate",0), errors="coerce").fillna(0),
        "billable": col("billable", True).fillna(True),
        "client": col("client","").astype(str),
        "invoice_amount": pd.to_numeric(col("invoice_amount",0), errors="coerce").fillna(0),
        "invoice_date": pd.to_datetime(col("invoice_date"), errors='coerce'),
        "paid_date": pd.to_datetime(col("paid_date"), errors='coerce'),
    })
    norm["revenue"] = np.where(norm["billable"], norm["hours"]*norm["rate"], norm["invoice_amount"])
    return norm.dropna(subset=["date"])

def normalize_saas(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith((".xlsx",".xls")):
        df = pd.read_excel(BytesIO(file_bytes))
    else:
        df = pd.read_csv(BytesIO(file_bytes))
    mapping = infer_columns(df, {
        "customer_id": ["customer_id","client_id","account_id"],
        "plan": ["plan","product","tier","package"],
        "start": ["start","start_date","activated_at","signup_date"],
        "end": ["end","end_date","cancelled_at","churned_at"],
        "mrr": ["mrr","monthly_recurring_revenue","amount"],
        "event_type": ["event_type","type"],
        "event_ts": ["ts","timestamp","event_ts","event_time","created_at"],
        "cac": ["cac","acquisition_cost"],
    })
    def col(key, default=None):
        c = mapping.get(key)
        return df[c] if c and c in df.columns else pd.Series([default]*len(df))
    norm = pd.DataFrame({
        "customer_id": col("customer_id","").astype(str),
        "plan": col("plan","").astype(str),
        "start": pd.to_datetime(col("start"), errors='coerce'),
        "end": pd.to_datetime(col("end"), errors='coerce'),
        "mrr": pd.to_numeric(col("mrr",0), errors="coerce").fillna(0),
        "event_type": col("event_type","").astype(str),
        "event_ts": pd.to_datetime(col("event_ts"), errors='coerce'),
        "cac": pd.to_numeric(col("cac", np.nan), errors="coerce"),
    })
    return norm

def wow_growth(daily_df: pd.DataFrame):
    if daily_df is None or len(daily_df) < 14:
        return None
    a = daily_df.sort_values(daily_df.columns[0])
    last = a.iloc[-7:,1].sum()
    prev = a.iloc[-14:-7,1].sum()
    return float(((last/prev)-1)*100) if prev>0 else None

def abc_pareto(df: pd.DataFrame):
    sk = df.groupby('sku', as_index=False)['revenue'].sum().sort_values('revenue', ascending=False)
    if sk.empty:
        sk["cum_share"] = []
        sk["ABC"] = []
        return sk
    sk['cum_rev'] = sk['revenue'].cumsum()
    total = sk['revenue'].sum()
    sk['cum_share'] = sk['cum_rev']/total if total>0 else 0
    def tag(p):
        if p <= 0.8: return 'A'
        if p <= 0.95: return 'B'
        return 'C'
    sk['ABC'] = sk['cum_share'].apply(tag)
    return sk

def anomalies_zscore(daily: pd.DataFrame, window=7, z_thresh=2.0):
    if daily.empty: return pd.DataFrame(columns=['order_date','revenue','z'])
    x = daily.copy().sort_values(daily.columns[0])
    roll_mean = x.iloc[:,1].rolling(window).mean()
    roll_std = x.iloc[:,1].rolling(window).std().replace(0, np.nan)
    z = (x.iloc[:,1]-roll_mean)/roll_std
    x['z'] = z
    return x[(z.abs()>z_thresh) & z.notna()]

def rfm(df: pd.DataFrame):
    d = df.copy()
    d['order_date'] = pd.to_datetime(d['order_date'], errors='coerce')
    d = d.dropna(subset=['order_date'])
    today = d['order_date'].max() + pd.Timedelta(days=1)
    cust = d.groupby('customer_id').agg(
        last_date=('order_date', 'max'),
        freq=('customer_id','count'),
        monetary=('revenue','sum')
    ).reset_index()
    cust['recency'] = (today - cust['last_date']).dt.days
    cust['R'] = pd.qcut(cust['recency'].rank(method='first'), 5, labels=[5,4,3,2,1]).astype(int)
    cust['F'] = pd.qcut(cust['freq'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    cust['M'] = pd.qcut(cust['monetary'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    def segment(row):
        if row['R']>=4 and row['F']>=4: return 'Champions'
        if row['R']>=4 and row['F']<=2: return 'New Customers'
        if row['R']<=2 and row['F']>=4: return 'Loyal but Aging'
        if row['R']<=2 and row['F']<=2: return 'At Risk'
        return 'Potential Loyalists'
    cust['Segment'] = cust.apply(segment, axis=1)
    return cust

def cohort_retention(df: pd.DataFrame):
    x = df.copy()
    x['order_date'] = pd.to_datetime(x['order_date'], errors='coerce')
    x = x.dropna(subset=['order_date'])
    x['order_month'] = x['order_date'].dt.to_period('M')
    cohort = x.groupby('customer_id')['order_month'].min().rename('cohort')
    x = x.join(cohort, on='customer_id')
    x['m_index'] = ((x['order_month'].dt.year - x['cohort'].dt.year) * 12 +
                    (x['order_month'].dt.month - x['cohort'].dt.month))
    pivot = x.pivot_table(index='cohort', columns='m_index',
                          values='customer_id', aggfunc='nunique').fillna(0)
    if pivot.shape[1] == 0:
        return pivot
    cohort_size = pivot.iloc[:, 0].replace(0, np.nan)
    ret = pivot.divide(cohort_size, axis=0).fillna(0).round(3)
    return ret

def price_volume_mix(df: pd.DataFrame):
    x = df.copy()
    x['order_date'] = pd.to_datetime(x['order_date'], errors='coerce')
    x = x.dropna(subset=['order_date'])
    x['ym'] = x['order_date'].dt.to_period('M')
    m = x.groupby(['ym']).agg(qty=('qty','sum'), revenue=('revenue','sum'))
    m['price'] = m['revenue']/m['qty'].replace(0, np.nan)
    m = m.dropna(subset=['price']).sort_index()
    if len(m) < 2:
        return pd.DataFrame(columns=['period','price_effect','volume_effect','total_change'])
    out = []
    periods = m.index.to_list()
    for i in range(1, len(periods)):
        p0, p1 = periods[i-1], periods[i]
        price0, qty0, rev0 = m.loc[p0, ['price','qty','revenue']]
        price1, qty1, rev1 = m.loc[p1, ['price','qty','revenue']]
        price_effect  = (price1 - price0) * qty0
        volume_effect = (qty1 - qty0) * price0
        total_change  = rev1 - rev0
        out.append({'period': str(p1),
                    'price_effect': float(price_effect),
                    'volume_effect': float(volume_effect),
                    'total_change': float(total_change)})
    return pd.DataFrame(out)

def yoy_monthly(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors='coerce')
    d = d.dropna(subset=[date_col])
    d['ym'] = d[date_col].dt.to_period('M')
    m = d.groupby('ym')[value_col].sum().reset_index()
    m['year'] = m['ym'].dt.year
    m['month'] = m['ym'].dt.month
    pivot = m.pivot(index='month', columns='year', values=value_col).fillna(0)
    return pivot

def basket_pairs(df: pd.DataFrame, order_col='order_id', sku_col='sku', top_n=15):
    pairs = {}
    for _, g in df.groupby(order_col):
        items = sorted(set(g[sku_col].astype(str)))
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                pairs[(items[i], items[j])] = pairs.get((items[i], items[j]), 0) + 1
    if not pairs:
        return pd.DataFrame(columns=['item_a','item_b','count'])
    out = pd.DataFrame([{'item_a':a,'item_b':b,'count':c} for (a,b),c in pairs.items()]).sort_values('count', ascending=False).head(top_n)
    return out

def ar_aging(invoices: pd.DataFrame) -> pd.DataFrame:
    inv = invoices.copy()
    if 'invoice_date' not in inv or 'invoice_amount' not in inv:
        return pd.DataFrame(columns=['bucket','Amount'])
    inv['invoice_date'] = pd.to_datetime(inv['invoice_date'], errors='coerce')
    today = inv['invoice_date'].max() + pd.Timedelta(days=1)
    inv['paid_date'] = pd.to_datetime(inv.get('paid_date', pd.NaT), errors='coerce')
    inv['open_amt'] = np.where(inv['paid_date'].notna(), 0, inv['invoice_amount'])
    inv['age'] = (today - inv['invoice_date']).dt.days
    bins = [0,30,60,90,180,9999]
    labels = ['0-30','31-60','61-90','91-180','180+']
    inv['bucket'] = pd.cut(inv['age'], bins=bins, labels=labels, right=True, include_lowest=True)
    return inv.groupby('bucket')['open_amt'].sum().reindex(labels).reset_index(name='Amount')

def utilization_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['date'] = pd.to_datetime(d['date'], errors='coerce')
    d = d.dropna(subset=['date'])
    d['week'] = d['date'].dt.isocalendar().week.astype(int)
    pivot = d.pivot_table(index='consultant', columns='week', values='hours', aggfunc='sum').fillna(0)
    return pivot

def saas_mrr_movements(subs: pd.DataFrame) -> pd.DataFrame:
    s = subs.copy()
    s['start'] = pd.to_datetime(s['start'], errors='coerce')
    s['end'] = pd.to_datetime(s['end'], errors='coerce')
    s = s.dropna(subset=['start'])
    start = s['start'].min().to_period('M')
    end = (s['end'].dropna().max().to_period('M') if s['end'].notna().any() else pd.Timestamp.today().to_period('M'))
    months = pd.period_range(start=start, end=end, freq='M')
    rows = []
    prev_act = pd.DataFrame(columns=['customer_id','mrr'])
    for m in months:
        s_ts = m.start_time
        e_ts = (m+1).start_time
        active = s[(s['start'] < e_ts) & ((s['end'].isna()) | (s['end'] >= s_ts))][['customer_id','mrr']].groupby('customer_id')['mrr'].sum().reset_index()
        cur = active.set_index('customer_id')['mrr']
        prv = prev_act.set_index('customer_id')['mrr'] if not prev_act.empty else pd.Series(dtype=float)
        customers = set(cur.index).union(set(prv.index))
        new = sum(cur.get(c, 0) for c in customers if c not in prv.index)
        churn = sum(prv.get(c, 0) for c in customers if c not in cur.index)
        expansion = sum(max(cur.get(c,0)-prv.get(c,0), 0) for c in customers if c in cur.index and c in prv.index)
        contraction = sum(max(prv.get(c,0)-cur.get(c,0), 0) for c in customers if c in cur.index and c in prv.index)
        rows.append({'month': m.to_timestamp(), 'New': new, 'Expansion': expansion, 'Contraction': contraction, 'Churn': churn, 'Net': new+expansion-contraction-churn, 'MRR': cur.sum()})
        prev_act = active
    return pd.DataFrame(rows)
