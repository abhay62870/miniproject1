import streamlit as st
import numpy as np
import pandas as pd
import json, os, sqlite3, hashlib, binascii
from pathlib import Path
try:
    import bcrypt
    _BCRYPT_AVAILABLE = True
except Exception:
    bcrypt = None
    _BCRYPT_AVAILABLE = False
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except Exception:
    yf = None
    _YF_AVAILABLE = False


def safe_rerun():
    try:
        return st.experimental_rerun()
    except Exception:
        st.session_state["__rerun_requested__"] = not st.session_state.get("__rerun_requested__", False)


_PBKDF2_ITER = 100_000


def hash_password(p: str) -> str:
    if _BCRYPT_AVAILABLE:
        return bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()
    s = hashlib.sha256(os.urandom(32)).hexdigest()
    dk = hashlib.pbkdf2_hmac("sha256", p.encode(), s.encode(), _PBKDF2_ITER)
    return f"pbkdf2${s}${binascii.hexlify(dk).decode()}"


def verify_password(p: str, h: str) -> bool:
    if _BCRYPT_AVAILABLE and h and not h.startswith("pbkdf2$"):
        try:
            return bcrypt.checkpw(p.encode(), h.encode())
        except Exception:
            return False
    try:
        if not h.startswith("pbkdf2$"):
            return False
        _, s, hh = h.split("$", 2)
        dk = hashlib.pbkdf2_hmac("sha256", p.encode(), s.encode(), _PBKDF2_ITER)
        return binascii.hexlify(dk).decode() == hh
    except Exception:
        return False

# --------------------------------------------
# 1. Portfolio Environment
# --------------------------------------------
class PortfolioEnv:
    def __init__(self, n_assets=3, steps=30, prices=None):
        self.n_assets, self.steps = n_assets, steps
        self.t = 0
        if prices is None:
            self.prices = np.random.uniform(50, 150, size=(steps, n_assets))
        else:
            p = np.array(prices)
            if p.shape[0] < steps:
                p = np.vstack([p, np.repeat(p[-1:], steps - p.shape[0], axis=0)])
            self.prices = p[:steps, :n_assets]
        self.balance = 1000.0

    def reset(self):
        self.t = 0
        self.balance = 1000.0
        return np.concatenate(([self.balance], self.prices[self.t]))

    def step(self, action):
        self.t += 1
        a = np.abs(action)
        w = a / (a.sum() or 1)
        ratio = self.prices[self.t] / self.prices[self.t - 1]
        self.balance *= float(np.dot(w, ratio))
        return np.concatenate(([self.balance], self.prices[self.t])), self.balance - 1000.0, self.t >= self.steps - 1


# --------------------------------------------
# 2. Simple DRL Agent
# --------------------------------------------
class SimpleAgent:
    def __init__(self, n_assets):
        self.n_assets = n_assets

    def act(self, _):
        return np.random.uniform(-1, 1, self.n_assets)


# --------------------------------------------
# 3. Fraud Detection
# --------------------------------------------
def simulate_transactions(n=100):
    n = max(1, int(n))
    df = pd.DataFrame({
        "user_id": np.random.randint(1000, 1100, n),
        "amount": np.random.uniform(10, 5000, n),
        "time_gap": np.random.uniform(0.1, 60, n),
    })
    k = min(5, n)
    if k > 0:
        df.loc[np.random.choice(df.index, k, replace=False), "amount"] *= 10
    return df


# ----------------------------
# Market data helpers
# ----------------------------
SECTORS = {
    "Technology": ["AAPL", "MSFT", "GOOG", "INTC", "AMD"],
    "Finance": ["JPM", "BAC", "GS", "C", "MS"],
    "Healthcare": ["JNJ", "PFE", "MRK", "ABBV", "TMO"],
    "Energy": ["XOM", "CVX", "BP", "TOT", "COP"],
    "Consumer": ["AMZN", "WMT", "PG", "KO", "PEP"],
}


# Try to load a richer ticker catalog with company names from `tickers.json` if present
TICKER_CATALOG = {}
TICKER_NAME = {}
cf = Path(__file__).parent / "tickers.json"
if cf.exists():
    try:
        raw = json.loads(cf.read_text())
        for sector, rows in raw.items():
            TICKER_CATALOG[sector] = [(str(t).upper(), str(n)) for t, n in rows]
            for t, n in TICKER_CATALOG[sector]:
                TICKER_NAME[t] = n
    except Exception:
        pass
if not TICKER_CATALOG:
    for s, tickers in SECTORS.items():
        TICKER_CATALOG[s] = [(t, t) for t in tickers]
        for t in tickers:
            TICKER_NAME[t] = t


def generate_price_series(tickers, steps=30, use_real=False):
    tickers = list(tickers)
    n = max(1, len(tickers))
    if use_real and _YF_AVAILABLE:
        try:
            period_days = max(steps + 5, 30)
            data = yf.download(tickers, period=f"{period_days}d", interval="1d", progress=False, threads=False)
            close = data["Close"] if "Close" in data else data
            if isinstance(close, pd.Series):
                close = close.to_frame()
            close = close.dropna(how="all")
            if close.shape[0] >= steps:
                recent = close.tail(steps)
                recent = recent.loc[:, [c for c in tickers if c in recent.columns]]
                prices = [recent[c].values if c in recent.columns else np.random.uniform(50, 150, size=steps) for c in tickers]
                return np.vstack(prices).T
        except Exception:
            pass
    prices = np.zeros((steps, n), dtype=float)
    for i in range(n):
        s0 = np.random.uniform(50, 150)
        mu, sigma = 0.0005, 0.01
        series = [s0]
        for _ in range(1, steps):
            series.append(series[-1] * np.exp(np.random.normal(mu, sigma)))
        prices[:, i] = series
    return prices


def detect_fraud(df):
    mean_amt = df["amount"].mean()
    std_amt = df["amount"].std()
    df["fraud_flag"] = df["amount"] > mean_amt + 2 * std_amt
    frauds = df[df["fraud_flag"]]
    return frauds


TEST_MODE = os.environ.get("MINIPROJECT_TEST") == "1"


def run_app():
    st.set_page_config(page_title="DRL Portfolio & Fraud Detection", layout="centered")
    st.title("💹 DRL Portfolio & Fraud Detection")
    DB = Path(__file__).parent / "users.db"

    def c():
        con = sqlite3.connect(str(DB)); con.row_factory = sqlite3.Row; return con

    def ensure():
        conn = c(); conn.execute("""CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, password TEXT NOT NULL)"""); conn.commit(); conn.close()

    def get_pw(u):
        ensure(); conn = c(); r = conn.execute("SELECT password FROM users WHERE username=?", (u,)).fetchone(); conn.close(); return r[0] if r else None

    def add_user(u, p):
        try:
            ensure(); conn = c(); conn.execute("INSERT INTO users(username,password) VALUES(?,?)", (u, hash_password(p))); conn.commit(); conn.close(); return True
        except sqlite3.IntegrityError:
            return False

    if "logged_in" not in st.session_state:
        st.session_state.update({"logged_in": False, "username": ""})

    if not st.session_state["logged_in"]:
        st.sidebar.header("Login / Signup")
        u = st.sidebar.text_input("Username", key="u")
        p = st.sidebar.text_input("Password", type="password", key="p")
        if st.sidebar.button("Login"):
            h = get_pw(u)
            if h and verify_password(p, h):
                st.session_state.update({"logged_in": True, "username": u}); safe_rerun()
            else:
                st.sidebar.error("Invalid credentials")
        st.sidebar.markdown("---")
        su = st.sidebar.text_input("New user", key="su")
        sp = st.sidebar.text_input("New pass", type="password", key="sp")
        if st.sidebar.button("Create account"):
            if not su or not sp:
                st.sidebar.error("Enter username and password")
            elif add_user(su, sp):
                st.sidebar.success("Account created; you can log in now.")
            else:
                st.sidebar.error("Could not create user")
        st.info("Please log in from the sidebar to access the app.")
        return

    st.sidebar.write(f"Logged in: **{st.session_state['username']}**")
    if st.sidebar.button("Logout"):
        st.session_state.update({"logged_in": False, "username": ""}); safe_rerun()

    tab1, tab2 = st.tabs(["Portfolio", "Fraud"])
    with tab1:
        steps = st.slider("Steps", 10, 100, 30)
        runs = st.number_input("Runs", 1, 50, 5)
        sector = st.selectbox("Sector", list(TICKER_CATALOG.keys()))
        catalog = TICKER_CATALOG.get(sector, [])
        opts = [f"{t} — {n}" for t, n in catalog]
        sel = st.multiselect("Tickers", opts, default=opts[:3])
        use_real = st.checkbox("Use real data (yfinance)")
        if use_real and not _YF_AVAILABLE:
            st.warning("yfinance not available; using simulated prices")
        if st.button("Run Simulation"):
            tickers = [s.split(" — ")[0] for s in sel] if sel else [t for t, _ in catalog[:3]]
            prices = generate_price_series(tickers, steps=steps, use_real=use_real)
            env = PortfolioEnv(len(tickers), steps, prices)
            agent = SimpleAgent(len(tickers))
            vals = []
            for _ in range(runs):
                state = env.reset(); done = False
                while not done:
                    a = agent.act(state)
                    state, r, done = env.step(a)
                vals.append(env.balance)
            st.success(f"Avg final value: ₹{np.mean(vals):.2f}")
            try:
                dfp = pd.DataFrame(prices, columns=tickers); st.line_chart(dfp)
            except Exception:
                pass
            st.bar_chart(vals)
            st.metric("One-Score", f"{int(np.clip((np.mean(vals)-1000)/100,0,100))}/100", delta=f"₹{np.mean(vals)-1000:.2f}")

    with tab2:
        n = st.slider("Transactions", 50, 500, 100)
        if st.button("Generate & Detect Fraud"):
            tx = simulate_transactions(n)
            fr = detect_fraud(tx)
            st.info(f"Detected {len(fr)} suspicious transactions")
            st.dataframe(fr.head(10))
        # Additional transaction views and spending analysis
        if st.checkbox("Show transaction summary"):
            txs = simulate_transactions(n)
            st.write(txs.describe())

        if st.checkbox("Show spending analysis"):
            tx_sp = simulate_transactions(n)
            total_spent = tx_sp["amount"].sum()
            avg_spent = tx_sp["amount"].mean()
            median_spent = tx_sp["amount"].median()
            st.subheader("Spending Summary")
            st.write(f"- **Total spent:** ₹{total_spent:,.2f}")
            st.write(f"- **Average transaction:** ₹{avg_spent:,.2f}")
            st.write(f"- **Median transaction:** ₹{median_spent:,.2f}")

            top_users = tx_sp.groupby("user_id")["amount"].sum().sort_values(ascending=False).head(10)
            st.subheader("Top 10 Users by Total Spend")
            st.bar_chart(top_users)

            counts, edges = np.histogram(tx_sp["amount"], bins=20)
            ranges = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(counts))]
            hist_df = pd.DataFrame({"count": counts}, index=ranges)
            st.subheader("Transaction Amount Distribution")
            st.bar_chart(hist_df)

            tx_sp = tx_sp.reset_index(drop=True)
            tx_sp["cumulative"] = tx_sp["amount"].cumsum()
            st.subheader("Cumulative Spend")
            st.line_chart(tx_sp[["cumulative"]])


if __name__ == "__main__" and not TEST_MODE:
    run_app()

