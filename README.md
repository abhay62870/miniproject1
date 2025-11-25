DRL-Based Portfolio Optimization & Fraud Detection (Streamlit)

Quick start (Windows PowerShell):

1) Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3) Run the app (bind to all interfaces for LAN access):

```powershell
.\.venv\Scripts\streamlit.exe run .\miniproject.py --server.address 0.0.0.0 --server.port 9999
```

Open http://localhost:9999 in your browser, or http://<your-ip>:9999 from other devices on your LAN.

Default demo account:
- username: admin
- password: admin123

You can also sign up from the sidebar (demo persistent store: `users.json`).

Security notes:
- This demo uses a simple `users.json` file with bcrypt-hashed passwords. For production, use a proper auth provider (OAuth, OpenID), HTTPS, and secure storage.
- Do not expose the app publicly without a reverse proxy and TLS (e.g., nginx + certbot).

Running tests:

```powershell
# run tests in test mode so the Streamlit UI doesn't start
$env:MINIPROJECT_TEST = '1'
python -m pytest -q
```

Development notes:
- The Streamlit UI is guarded so importing `miniproject` in tests will not start the app when `MINIPROJECT_TEST=1`.
