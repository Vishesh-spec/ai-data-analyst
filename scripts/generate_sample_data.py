"""Generate sample datasets (with a few injected anomalies) into ./data."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
regions = ["North", "South", "East", "West"]
products = {"Laptop": 900, "Phone": 600, "Tablet": 350, "Headphones": 120, "Smartwatch": 220, "Cable": 15}
cust = pd.DataFrame({"customer_id": range(1, 41),
                     "customer_name": [f"Customer {i:02d}" for i in range(1, 41)],
                     "segment": rng.choice(["Consumer", "SMB", "Enterprise"], 40),
                     "country": rng.choice(["UK", "India", "USA", "Germany"], 40)})
n = 600
dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
prod = rng.choice(list(products), n, p=[.12, .18, .12, .2, .1, .28])
qty = rng.integers(1, 6, n)
price = np.array([products[p] for p in prod]) * rng.normal(1, .05, n)
sales = pd.DataFrame({"order_id": range(1000, 1000 + n), "order_date": dates.date,
                      "customer_id": rng.integers(1, 41, n), "region": rng.choice(regions, n),
                      "product": prod, "quantity": qty, "unit_price": price.round(2)})
sales["revenue"] = (sales.quantity * sales.unit_price).round(2)
sales.loc[[10, 250], ["quantity", "revenue"]] = [[250, 250 * 900], [180, 180 * 600]]  # anomalies
sales.loc[77, "unit_price"] = 9.99; sales.loc[77, "revenue"] = 9.99 * sales.loc[77, "quantity"]
sales.loc[[5, 99], "region"] = np.nan  # data quality issue
sales.sort_values("order_date").to_csv("data/sales.csv", index=False)
cust.to_csv("data/customers.csv", index=False)
print("written data/sales.csv, data/customers.csv")
