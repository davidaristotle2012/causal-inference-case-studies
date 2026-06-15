import numpy as np
import pandas as pd


def generate_sales_data(
    seed: int = 42,
    n_customers: int = 1000,
    start_date: str = "2024-01-01",
    end_date: str = "2026-12-31",
    max_mean: float = 1_000_000,
    min_mean: float = 1_000,
    power: float = 2.5,
    top1_target_share: float = 0.30,
    brand_base_props: tuple = (0.1, 0.3, 0.6),
    noise_sd: float = 0.05,
    discount_min: float = 0.0,
    discount_max: float = 5.0,
    remove_frac: float = 0.10,
) -> pd.DataFrame:
    """
    Generate sales order line item table with columns:
    customer_key, product_key, product_name, revenue, sales_date, discount_pcnt.
    """
    np.random.seed(seed)
    # customers and skewed annual means
    customers = pd.DataFrame({"customer_key": np.arange(1, n_customers + 1)})
    ranks = np.arange(1, n_customers + 1)
    scaled = (n_customers + 1 - ranks) ** power
    scaled = (scaled - scaled.min()) / (scaled.max() - scaled.min())
    annual_means = min_mean + scaled * (max_mean - min_mean)
    customers["annual_mean_revenue"] = annual_means

    # scale so top 1% ~ target share of total
    total = customers["annual_mean_revenue"].sum()
    top_n = max(1, int(n_customers * 0.01))
    top1pct = customers.nlargest(top_n, "annual_mean_revenue")[
        "annual_mean_revenue"
    ].sum()
    target_top1 = top1_target_share * total
    scale_factor = target_top1 / top1pct if top1pct > 0 else 1.0
    customers["annual_mean_revenue"] *= scale_factor

    # brand proportions per outlet with small noise (ensure sum to 1)
    base_props = np.array(brand_base_props)

    def noisy_props():
        noise = np.random.normal(0, noise_sd, size=base_props.shape)
        p = base_props + noise
        p = np.clip(p, 0.001, None)
        return p / p.sum()

    props = np.vstack([noisy_props() for _ in range(n_customers)])
    customers[["p_fanta", "p_sprite", "p_coke"]] = props

    # dates (daily)
    dates = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D")

    brands = ["Fanta", "Sprite", "Coke"]
    product_key_map = {"Fanta": 1, "Sprite": 2, "Coke": 3}
    rows = []

    for _, cust_row in customers.iterrows():
        cust = int(cust_row["customer_key"])
        daily_mean = cust_row["annual_mean_revenue"] / 365.0
        customer_multiplier = np.random.gamma(shape=2.0, scale=1.0)
        p_f, p_s, p_c = cust_row[["p_fanta", "p_sprite", "p_coke"]].values

        for d in dates:
            # skewed daily revenue via Gamma
            daily_rev = np.random.gamma(
                shape=0.5, scale=daily_mean * customer_multiplier
            )
            # allocate to brands using multinomial on rounded units (if any)
            units = int(np.round(daily_rev)) if daily_rev >= 1 else 0
            alloc = (
                np.random.multinomial(units, [p_f, p_s, p_c])
                if units > 0
                else np.array([0, 0, 0])
            )
            prices = np.random.normal([1.0, 1.0, 1.0], 0.05)
            revs = alloc * prices

            discount_pcnt = float(np.random.uniform(discount_min, discount_max))

            for b_name, rev in zip(brands, revs):
                if rev > 0:
                    rows.append(
                        {
                            "customer_key": cust,
                            "product_key": product_key_map[b_name],
                            "product_name": b_name,
                            "revenue": float(rev),
                            "sales_date": d,
                            "discount_pcnt": discount_pcnt,
                        }
                    )

    df_sales = pd.DataFrame(rows)

    # randomly remove a fraction of rows (zero-sale days)
    if not df_sales.empty and remove_frac > 0:
        n_remove = int(remove_frac * len(df_sales))
        remove_idx = np.random.choice(df_sales.index, size=n_remove, replace=False)
        df_sales = df_sales.drop(index=remove_idx).reset_index(drop=True)

    return df_sales


def generate_weekcover_data(
    seed: int = 42,
    n_customers: int = 1000,
    start_date: str = "2024-01-01",
    end_date: str = "2026-12-31",
    weekcover_mean: float = 2.0,
    weekcover_sd: float = 3.0,
) -> pd.DataFrame:
    cust = pd.DataFrame({"customer_key": np.arange(1, n_customers + 1)})
    dates = pd.DataFrame(
        {
            "week_start": pd.date_range(
                pd.Timestamp(start_date), pd.Timestamp(end_date), freq="W-MON"
            )
        }
    )
    product_keys = pd.DataFrame({"product_key": [1, 2, 3]})

    cross = cust.merge(dates, how="cross").merge(product_keys, how="cross")

    cust_shifts = np.random.normal(1.0, 0.5, size=n_customers)
    cross["cust_shift"] = cross["customer_key"].map(lambda x: cust_shifts[x - 1])

    rng = np.random.default_rng(seed)
    noises = rng.normal(
        loc=weekcover_mean * cross["cust_shift"], scale=weekcover_sd, size=len(cross)
    )
    cross["weekcover"] = np.clip(noises, 0, 20).astype(float)

    return cross[["customer_key", "week_start", "product_key", "weekcover"]]
