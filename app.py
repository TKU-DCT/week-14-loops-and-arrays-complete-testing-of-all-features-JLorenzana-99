import hashlib
import platform
import sqlite3
import subprocess
import time
from datetime import datetime, timedelta

import pandas as pd
import psutil
import streamlit as st

# --- Constants and Session Defaults ---
DB_NAME = "log.db"
MIN_SAMPLE_SECONDS = 5


st.set_page_config(
    page_title="System Monitor Dashboard",
    page_icon="PC",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Session state defaults (kept here so rest of the code can rely on them)
st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("username", None)
st.session_state.setdefault("role", None)
st.session_state.setdefault("dark_mode", True)
st.session_state.setdefault("auto_refresh", False)
st.session_state.setdefault("refresh_interval", 30)
st.session_state.setdefault("cpu_threshold", 80)
st.session_state.setdefault("memory_threshold", 85)
st.session_state.setdefault("disk_threshold", 90)
st.session_state.setdefault("last_sample_at", None)


def apply_theme():
    """Inject a light/dark theme based on the toggle."""
    # Palette chosen to improve contrast and add subtle depth to cards/components.
    dark = st.session_state.dark_mode
    bg = "#0D1117" if dark else "#F6F8FB"
    text = "#E6EDF3" if dark else "#0F172A"
    panel = "#161B22" if dark else "#FFFFFF"
    card = "#1C2128" if dark else "#FFFFFF"
    accent = "#3B82F6" if dark else "#1D4ED8"
    subtle = "#9CA3AF" if dark else "#6B7280"
    border = "#30363D" if dark else "#E5E7EB"
    shadow = "0 8px 24px rgba(0,0,0,0.25)" if dark else "0 10px 30px rgba(15,23,42,0.08)"

    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {bg};
            --text: {text};
            --panel: {panel};
            --card: {card};
            --accent: {accent};
            --subtle: {subtle};
            --border: {border};
            --shadow: {shadow};
        }}

        .stApp, body {{
            background-color: var(--bg);
            color: var(--text);
        }}
        .stSidebar {{
            background-color: var(--panel);
            border-right: 1px solid var(--border);
            box-shadow: var(--shadow);
        }}
        .stSidebar, .stSidebar * {{
            color: var(--text) !important;
        }}
        h1, h2, h3, h4, h5, h6, p, span, label {{
            color: var(--text) !important;
        }}
        .stMetric, .stMarkdown, .stText {{
            color: var(--text) !important;
        }}
        div[data-testid="stMetricValue"], div[data-testid="stMetricDelta"] {{
            color: var(--text) !important;
        }}
        .stButton>button {{
            background: linear-gradient(135deg, {accent}, #2563EB);
            color:#FFFFFF;
            border-radius:6px;
            border: none;
            box-shadow: var(--shadow);
            transition: transform 0.08s ease, box-shadow 0.12s ease;
        }}
        .stButton>button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 12px 30px rgba(0,0,0,0.25);
        }}
        [data-testid="stHeader"], [data-testid="stToolbar"] {{
            background: var(--bg);
        }}
        [data-testid="stExpander"] {{
            background: var(--card);
            color: var(--text);
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
        }}
        .stDataFrame, .stTable {{
            background: var(--card);
            color: var(--text);
            border-radius: 8px;
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
        }}
        .stCaption, .stAlert {{
            color: var(--subtle);
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 16px;
            color: var(--text);
        }}
        .css-1d391kg, .block-container {{
            padding-top: 12px;
        }}
        /* Card look for metrics containers */
        div[data-testid="metric-container"] {{
            background: var(--card);
            padding: 12px 16px;
            border-radius: 10px;
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_theme()


# --- Database / Persistence ---
def init_db():
    """Ensure the SQLite database and required tables exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS system_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            cpu REAL,
            memory REAL,
            disk REAL,
            ping_status TEXT,
            ping_ms REAL
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            alert_type TEXT,
            value REAL,
            threshold REAL,
            message TEXT
        )
        """
    )
    conn.commit()

    # Users table for auth/roles
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin','user'))
        )
        """
    )
    conn.commit()
    seed_default_users(cursor)

    conn.close()


def seed_default_users(cursor):
    """Create default users if table is empty."""
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    if count == 0:
        users = [
            ("admin", hash_password("admin123"), "admin"),
            ("user", hash_password("user123"), "user"),
        ]
        cursor.executemany(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", users
        )
        cursor.connection.commit()


# --- Networking / Metric collection helpers ---
def parse_ping_time(output: str) -> float:
    """Extract ping time in milliseconds from ping output; return -1 on failure."""
    for line in output.splitlines():
        lower = line.lower()
        if "time=" in lower or "time<" in lower:
            separator = "time=" if "time=" in lower else "time<"
            try:
                time_str = lower.split(separator, 1)[1].split()[0].replace("ms", "").strip()
                return float(time_str)
            except (ValueError, IndexError):
                return -1
    return -1


def ping_host(host: str = "8.8.8.8") -> tuple[str, float]:
    """Ping a host and return (status, ms)."""
    try:
        param = "-n" if platform.system().lower() == "windows" else "-c"
        output = subprocess.check_output(
            ["ping", param, "1", host], stderr=subprocess.DEVNULL
        ).decode()
        return "UP", parse_ping_time(output)
    except Exception:
        return "DOWN", -1.0


def get_system_info() -> tuple[str, float, float, float, str, float]:
    """Collect current system metrics."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cpu = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory().percent
    disk = psutil.disk_usage("/").percent
    ping_status, ping_ms = ping_host()
    return now, cpu, memory, disk, ping_status, ping_ms


def insert_log(data):
    """Insert one row into system_log."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO system_log (timestamp, cpu, memory, disk, ping_status, ping_ms)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        data,
    )
    conn.commit()
    conn.close()


def insert_alert(timestamp, alert_type, value, threshold, message):
    """Insert alert record into alerts_log."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO alerts_log (timestamp, alert_type, value, threshold, message)
        VALUES (?, ?, ?, ?, ?)
        """,
        (timestamp, alert_type, value, threshold, message),
    )
    conn.commit()
    conn.close()


def check_alerts(data, thresholds):
    """Check metrics against thresholds and record alerts."""
    timestamp, cpu, memory, disk, ping_status, ping_ms = data
    alerts = []

    if cpu > thresholds["cpu"]:
        msg = f"ALERT: High CPU usage {cpu:.1f}% (> {thresholds['cpu']}%)"
        alerts.append(("CPU", cpu, thresholds["cpu"], msg))

    if cpu > 90:
        email_msg = f"EMAIL ALERT: CPU usage reached {cpu:.1f}% (> 90%)"
        alerts.append(("EMAIL", cpu, 90, email_msg))

    if memory > thresholds["memory"]:
        msg = f"ALERT: High Memory usage {memory:.1f}% (> {thresholds['memory']}%)"
        alerts.append(("MEMORY", memory, thresholds["memory"], msg))

    if disk > thresholds["disk"]:
        msg = f"ALERT: High Disk usage {disk:.1f}% (> {thresholds['disk']}%)"
        alerts.append(("DISK", disk, thresholds["disk"], msg))

    if ping_status == "DOWN":
        msg = "ALERT: Ping failed (host unreachable)"
        alerts.append(("PING", ping_ms, 0, msg))

    for alert_type, value, threshold, message in alerts:
        insert_alert(timestamp, alert_type, value, threshold, message)

    return alerts


# --- Sampling and throttling ---
def collect_metrics(thresholds, min_interval_seconds):
    """
    Collect one sample if enough time has passed.
    Returns (data, alerts). If throttled, both are None/[].
    """
    last_sample = st.session_state.get("last_sample_at")
    now = datetime.now()
    if last_sample:
        try:
            last_dt = datetime.fromisoformat(last_sample)
            if (now - last_dt).total_seconds() < min_interval_seconds:
                return None, []
        except ValueError:
            pass

    data = get_system_info()
    insert_log(data)
    alerts = check_alerts(data, thresholds)
    st.session_state.last_sample_at = now.isoformat()
    st.cache_data.clear()  # ensure fresh reads
    return data, alerts


# --- Read APIs (cached for the dashboard) ---
@st.cache_data(ttl=10)
def get_system_logs(
    ping_filter=None,
    date_filter=None,
    cpu_threshold=None,
    memory_threshold=None,
    disk_threshold=None,
):
    """Fetch system logs with optional filters."""
    conn = sqlite3.connect(DB_NAME)
    conditions = []
    params = []

    if ping_filter and ping_filter != "All":
        conditions.append("ping_status = ?")
        params.append(ping_filter)

    if date_filter:
        start_date, end_date = date_filter
        conditions.append("DATE(timestamp) BETWEEN ? AND ?")
        params.extend([start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")])

    if cpu_threshold is not None:
        conditions.append("cpu >= ?")
        params.append(cpu_threshold)

    if memory_threshold is not None:
        conditions.append("memory >= ?")
        params.append(memory_threshold)

    if disk_threshold is not None:
        conditions.append("disk >= ?")
        params.append(disk_threshold)

    where_clause = " AND ".join(conditions) if conditions else "1=1"
    query = f"SELECT * FROM system_log WHERE {where_clause} ORDER BY id DESC"

    df = pd.read_sql_query(query, conn, params=params if params else None)
    conn.close()
    return df


@st.cache_data(ttl=10)
def get_alerts_log():
    """Fetch recent alerts."""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query(
        "SELECT * FROM alerts_log ORDER BY id DESC LIMIT 50", conn
    )
    conn.close()
    return df


@st.cache_data(ttl=10)
def get_statistics(thresholds_tuple):
    """Return quick stats using the current thresholds."""
    cpu_th, mem_th, disk_th = thresholds_tuple
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM system_log")
    log_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM alerts_log")
    alert_count = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT cpu, memory, disk, ping_status, ping_ms
        FROM system_log
        ORDER BY id DESC
        LIMIT 1
        """
    )
    latest = cursor.fetchone()

    cursor.execute(
        """
        SELECT COUNT(*) FROM system_log
        WHERE cpu > ? OR memory > ? OR disk > ?
        """,
        (cpu_th, mem_th, disk_th),
    )
    threshold_violations = cursor.fetchone()[0]

    conn.close()
    return {
        "log_count": log_count,
        "alert_count": alert_count,
        "threshold_violations": threshold_violations,
        "latest_cpu": latest[0] if latest else 0,
        "latest_memory": latest[1] if latest else 0,
        "latest_disk": latest[2] if latest else 0,
        "latest_ping_status": latest[3] if latest else "N/A",
        "latest_ping_ms": latest[4] if latest else 0,
    }


@st.cache_data(ttl=15)
def get_summary_metrics():
    """
    Return aggregate metrics:
    - average CPU / memory / disk
    - average ping (only successful pings)
    - alert count using fixed thresholds CPU>80, MEM>85, DISK>90
    - full dataframe for downstream use
    """
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM system_log ORDER BY id DESC", conn)
    conn.close()
    if df.empty:
        return None

    avg_cpu = df["cpu"].mean()
    avg_memory = df["memory"].mean()
    avg_disk = df["disk"].mean()
    successful_pings = df[df["ping_ms"] > 0]["ping_ms"]
    avg_ping = successful_pings.mean() if not successful_pings.empty else None

    alert_mask = (df["cpu"] > 80) | (df["memory"] > 85) | (df["disk"] > 90)
    alert_count = int(alert_mask.sum())

    return {
        "df": df,
        "avg_cpu": avg_cpu,
        "avg_memory": avg_memory,
        "avg_disk": avg_disk,
        "avg_ping": avg_ping,
        "alert_count": alert_count,
    }


@st.cache_data(ttl=10)
def get_recent_alerts(hours: int = 24):
    """Fetch alerts within the last N hours."""
    since_ts = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query(
        "SELECT * FROM alerts_log WHERE timestamp >= ? ORDER BY timestamp DESC",
        conn,
        params=[since_ts],
    )
    conn.close()
    return df


# --- Derived metrics & summary builders ---
@st.cache_data(ttl=10)
def count_cpu_exceedances(threshold: float = 80.0) -> int:
    """Return how many samples crossed the CPU threshold."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM system_log WHERE cpu > ?", (threshold,))
    count = cursor.fetchone()[0]
    conn.close()
    return int(count)


def generate_text_summary(summary, cpu_exceed_count: int, alerts_df):
    """Build a human-friendly text summary using loops and string formatting."""
    if not summary:
        return "No data available yet. Add samples to generate a summary."

    lines = [
        f"System Summary generated at {datetime.now():%Y-%m-%d %H:%M:%S}",
        "-" * 50,
        f"Average CPU usage: {summary['avg_cpu']:.1f}%",
        f"Average Memory usage: {summary['avg_memory']:.1f}%",
        f"Average Disk usage: {summary['avg_disk']:.1f}%",
        (
            f"Average Ping: {summary['avg_ping']:.1f} ms"
            if summary["avg_ping"] is not None
            else "Average Ping: N/A (no successful pings)"
        ),
        f"CPU samples above 80%: {cpu_exceed_count}",
        f"Alerts (CPU>80/MEM>85/DISK>90): {summary['alert_count']}",
        "",
    ]

    lines.append("Recent alerts:")
    if alerts_df is not None and not alerts_df.empty:
        for _, row in alerts_df.head(5).iterrows():
            lines.append(f"- {row['timestamp']} | {row['alert_type']}: {row['message']}")
    else:
        lines.append("- No alerts found in history.")

    return "\n".join(lines)


def save_summary_to_file(summary_text: str, prefix: str = "system_summary") -> str:
    """Write the provided summary to a timestamped text file and return its name."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{prefix}_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(summary_text)
    return filename


def export_summary_pdf(summary):
    """
    Create a simple PDF summary using fpdf if available.
    Returns bytes or None if fpdf is missing.
    """
    try:
        from fpdf import FPDF  # type: ignore
    except Exception:
        return None

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "System Monitor Summary", ln=1)
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, f"Average CPU: {summary['avg_cpu']:.2f}%", ln=1)
    pdf.cell(0, 10, f"Average Memory: {summary['avg_memory']:.2f}%", ln=1)
    pdf.cell(0, 10, f"Average Disk: {summary['avg_disk']:.2f}%", ln=1)
    ping_line = (
        f"Average Ping: {summary['avg_ping']:.2f} ms" if summary["avg_ping"] is not None else "Average Ping: N/A"
    )
    pdf.cell(0, 10, ping_line, ln=1)
    pdf.cell(0, 10, f"Alerts (CPU>80/MEM>85/DISK>90): {summary['alert_count']}", ln=1)

    return pdf.output(dest="S").encode("latin-1")


# --- Authentication ---
def hash_password(password):
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def get_user(username):
    """Fetch user record by username."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT username, password_hash, role FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"username": row[0], "password_hash": row[1], "role": row[2]}
    return None


def verify_credentials(username, password):
    """Verify login credentials against the users table."""
    user = get_user(username)
    if not user:
        return None
    if user["password_hash"] == hash_password(password):
        return user
    return None


# --- UI: Authentication ---
def login_page():
    """Display login page."""
    st.markdown(
        """
        <div style='text-align: center; padding: 40px 0 20px 0;'>
            <h1 style='font-size: 42px; margin: 0;'>System Monitor</h1>
            <p style='color: gray;'>Secure Login Portal</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Sign In")
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Login", use_container_width=True)
            if submitted:
                user = verify_credentials(username, password) if username and password else None
                if user:
                    st.session_state.logged_in = True
                    st.session_state.username = user["username"]
                    st.session_state.role = user["role"]
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        st.info("Default credentials - admin/admin123 (admin) or user/user123 (user)")


def logout():
    """Logout and reset session."""
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None
    st.rerun()


# --- UI: Dashboard ---
def dashboard_page(new_alerts):
    """Main dashboard page."""
    st.title("System Monitor Dashboard")
    if new_alerts:
        for _, _, _, msg in new_alerts:
            st.toast(msg)
    st.markdown("---")

    st.subheader("Filters")
    filter_col1, filter_col2, filter_col3, filter_col4, filter_col5, filter_col6 = st.columns(6)
    with filter_col1:
        ping_filter = st.selectbox("Ping Status", ["All", "UP", "DOWN"])
    with filter_col2:
        cpu_filter = st.slider("CPU Filter (%)", 0, 100, 0, 5)
    with filter_col3:
        memory_filter = st.slider("Memory Filter (%)", 0, 100, 0, 5)
    with filter_col4:
        disk_filter = st.slider("Disk Filter (%)", 0, 100, 0, 5)
    with filter_col5:
        use_date_filter = st.checkbox("Enable Date Filter")
    with filter_col6:
        num_records = st.slider("Records to Display", 5, 100, 10)

    date_filter = None
    if use_date_filter:
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
        with date_col2:
            end_date = st.date_input("End Date", datetime.now())
        date_filter = (start_date, end_date)

    st.markdown("---")

    thresholds_tuple = (
        st.session_state.cpu_threshold,
        st.session_state.memory_threshold,
        st.session_state.disk_threshold,
    )

    try:
        stats = get_statistics(thresholds_tuple)
        summary = get_summary_metrics()
        alerts_df = get_alerts_log()
        cpu_exceed_count = count_cpu_exceedances(80)
        st.header("Current System Status")
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            cpu_delta = stats["latest_cpu"] - st.session_state.cpu_threshold
            st.metric(
                "CPU Usage",
                f"{stats['latest_cpu']:.1f}%",
                delta=f"{cpu_delta:.1f}%" if stats["latest_cpu"] > st.session_state.cpu_threshold else None,
                delta_color="inverse",
            )

        with col2:
            mem_delta = stats["latest_memory"] - st.session_state.memory_threshold
            st.metric(
                "Memory Usage",
                f"{stats['latest_memory']:.1f}%",
                delta=f"{mem_delta:.1f}%" if stats["latest_memory"] > st.session_state.memory_threshold else None,
                delta_color="inverse",
            )

        with col3:
            disk_delta = stats["latest_disk"] - st.session_state.disk_threshold
            st.metric(
                "Disk Usage",
                f"{stats['latest_disk']:.1f}%",
                delta=f"{disk_delta:.1f}%" if stats["latest_disk"] > st.session_state.disk_threshold else None,
                delta_color="inverse",
            )

        with col4:
            st.metric("Ping Status", f"{stats['latest_ping_status']}")

        with col5:
            ping_display = f"{stats['latest_ping_ms']:.1f} ms" if stats["latest_ping_ms"] > 0 else "N/A"
            st.metric("Ping Time", ping_display)

        with col6:
            st.metric(
                "Alert Count",
                stats["threshold_violations"],
                delta=f"{stats['threshold_violations']}" if stats["threshold_violations"] > 0 else "0",
                delta_color="inverse",
            )

        st.markdown("---")

        st.header("Key Averages & Reports")
        if summary:
            avg_ping_display = f"{summary['avg_ping']:.1f} ms" if summary["avg_ping"] is not None else "N/A"
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Avg CPU", f"{summary['avg_cpu']:.1f}%")
            c2.metric("Avg Memory", f"{summary['avg_memory']:.1f}%")
            c3.metric("Avg Disk", f"{summary['avg_disk']:.1f}%")
            c4.metric("Avg Ping", avg_ping_display)
            c5.metric("Alerts (CPU>80/MEM>85/DISK>90)", summary["alert_count"])
            c6.metric("CPU > 80% Count", cpu_exceed_count)

            report_df = pd.DataFrame(
                [
                    {
                        "avg_cpu": summary["avg_cpu"],
                        "avg_memory": summary["avg_memory"],
                        "avg_disk": summary["avg_disk"],
                        "avg_ping": summary["avg_ping"] if summary["avg_ping"] is not None else -1,
                        "alert_count": summary["alert_count"],
                    }
                ]
            )
            col_csv, col_pdf = st.columns(2)
            with col_csv:
                st.download_button(
                    "Download Report (CSV)",
                    data=report_df.to_csv(index=False),
                    file_name="system_summary.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with col_pdf:
                pdf_bytes = export_summary_pdf(summary)
                if pdf_bytes:
                    st.download_button(
                        "Download Report (PDF)",
                        data=pdf_bytes,
                        file_name="system_summary.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                else:
                    st.info("Install 'fpdf' to enable PDF export (e.g., pip install fpdf).")

            st.subheader("Text Summary & Export")
            with st.expander("Generate text summary and save", expanded=False):
                recent_for_summary = alerts_df.head(5) if alerts_df is not None and not alerts_df.empty else None
                summary_text = generate_text_summary(summary, cpu_exceed_count, recent_for_summary)
                st.text_area("Summary Preview", summary_text, height=240)
                txt_col1, txt_col2 = st.columns(2)
                with txt_col1:
                    st.download_button(
                        "Download Summary (.txt)",
                        data=summary_text,
                        file_name="system_summary.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                with txt_col2:
                    if st.button("Save summary to text file", use_container_width=True):
                        saved_file = save_summary_to_file(summary_text)
                        st.success(f"Summary saved to {saved_file}")
        else:
            st.info("No data yet to compute averages.")

        st.markdown("---")

        st.header("Recent Alerts")
        if not alerts_df.empty:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Total Alerts", stats["alert_count"])
            with st.expander(f"View Latest {min(20, len(alerts_df))} Alerts", expanded=False):
                for _, alert in alerts_df.head(20).iterrows():
                    st.warning(f"{alert['timestamp']} - {alert['message']}")

            email_alerts = alerts_df[alerts_df["alert_type"] == "EMAIL"]
            if not email_alerts.empty:
                latest_email = email_alerts.iloc[0]
                st.info(
                    f"Simulated email alerts sent: {len(email_alerts)} | Latest: {latest_email['timestamp']} - {latest_email['message']}"
                )
        else:
            st.success("No alerts triggered! All systems operating normally.")

        st.markdown("---")

        st.header("System Logs")
        df = get_system_logs(
            ping_filter if ping_filter != "All" else None,
            date_filter,
            cpu_filter if cpu_filter > 0 else None,
            memory_filter if memory_filter > 0 else None,
            disk_filter if disk_filter > 0 else None,
        )

        if not df.empty:
            st.info(
                f"Showing {min(num_records, len(df))} of {len(df)} filtered records (Total: {stats['log_count']})"
            )
            display_df = df.head(num_records).copy()
            display_df["cpu"] = display_df["cpu"].apply(lambda x: f"{x:.1f}%")
            display_df["memory"] = display_df["memory"].apply(lambda x: f"{x:.1f}%")
            display_df["disk"] = display_df["disk"].apply(lambda x: f"{x:.1f}%")
            display_df["ping_ms"] = display_df["ping_ms"].apply(
                lambda x: f"{x:.1f} ms" if x > 0 else "N/A"
            )
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No data available matching the current filters.")
            return

        st.markdown("---")

        st.header("Performance Charts")
        chart_df = df.sort_values("id")
        if not chart_df.empty:
            chart_data = chart_df[["timestamp", "cpu", "memory", "disk"]].copy()
            chart_data = chart_data.rename(columns={"cpu": "CPU %", "memory": "Memory %", "disk": "Disk %"})
            chart_data["timestamp"] = pd.to_datetime(chart_data["timestamp"])
            chart_data = chart_data.set_index("timestamp")

            st.subheader("System Resource Usage Over Time")
            st.line_chart(chart_data, height=400)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"CPU Threshold: {st.session_state.cpu_threshold}%")
            with col2:
                st.caption(f"Memory Threshold: {st.session_state.memory_threshold}%")
            with col3:
                st.caption(f"Disk Threshold: {st.session_state.disk_threshold}%")

            st.markdown("---")

            st.subheader("Network Ping Response Time")
            ping_df = chart_df[chart_df["ping_ms"] > 0].copy()
            if not ping_df.empty:
                ping_chart = ping_df[["timestamp", "ping_ms"]].copy()
                ping_chart["timestamp"] = pd.to_datetime(ping_chart["timestamp"])
                ping_chart = ping_chart.set_index("timestamp")
                ping_chart = ping_chart.rename(columns={"ping_ms": "Ping (ms)"})
                st.line_chart(ping_chart, height=300, color="#6C5CE7")
            else:
                st.info("No successful ping data available for current filters.")

            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Average Resource Usage")
                avg_stats = pd.DataFrame(
                    {
                        "Metric": ["CPU", "Memory", "Disk"],
                        "Average (%)": [
                            f"{chart_df['cpu'].mean():.2f}",
                            f"{chart_df['memory'].mean():.2f}",
                            f"{chart_df['disk'].mean():.2f}",
                        ],
                        "Max (%)": [
                            f"{chart_df['cpu'].max():.2f}",
                            f"{chart_df['memory'].max():.2f}",
                            f"{chart_df['disk'].max():.2f}",
                        ],
                        "Min (%)": [
                            f"{chart_df['cpu'].min():.2f}",
                            f"{chart_df['memory'].min():.2f}",
                            f"{chart_df['disk'].min():.2f}",
                        ],
                    }
                )
                st.dataframe(avg_stats, use_container_width=True, hide_index=True)

            with col2:
                st.subheader("Ping Status Summary")
                ping_counts = chart_df["ping_status"].value_counts()
                ping_summary = pd.DataFrame(
                    {
                        "Status": ping_counts.index,
                        "Count": ping_counts.values,
                        "Percentage": [f"{(count / len(chart_df) * 100):.1f}%" for count in ping_counts.values],
                    }
                )
                st.dataframe(ping_summary, use_container_width=True, hide_index=True)
        else:
            st.info("No data available for charts with current filters.")

        st.markdown("---")
        st.header("Alert History (Last 24 Hours)")
        recent_alerts = get_recent_alerts(hours=24)
        if recent_alerts.empty:
            st.success("No alerts in the last 24 hours.")
        else:
            st.dataframe(recent_alerts, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Make sure 'log.db' exists in the same directory.")


# --- UI: Configuration ---
def configuration_page():
    """Configuration page with threshold management."""
    st.title("Configuration Panel")
    st.markdown("---")

    st.subheader("Alert Threshold Settings")
    st.markdown("Adjust the thresholds for system alerts. Changes apply immediately.")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### CPU Threshold")
        cpu_threshold = st.slider(
            "CPU Usage Alert (%)",
            min_value=50,
            max_value=100,
            value=st.session_state.cpu_threshold,
            step=5,
            help="Alert when CPU usage exceeds this percentage",
        )
        st.session_state.cpu_threshold = cpu_threshold

        st.markdown("### Memory Threshold")
        memory_threshold = st.slider(
            "Memory Usage Alert (%)",
            min_value=50,
            max_value=100,
            value=st.session_state.memory_threshold,
            step=5,
            help="Alert when memory usage exceeds this percentage",
        )
        st.session_state.memory_threshold = memory_threshold

        st.markdown("### Disk Threshold")
        disk_threshold = st.slider(
            "Disk Usage Alert (%)",
            min_value=50,
            max_value=100,
            value=st.session_state.disk_threshold,
            step=5,
            help="Alert when disk usage exceeds this percentage",
        )
        st.session_state.disk_threshold = disk_threshold

    with col2:
        st.markdown("### Current Threshold Summary")
        st.info(
            f"""
            Active Thresholds
            - CPU: {st.session_state.cpu_threshold}%
            - Memory: {st.session_state.memory_threshold}%
            - Disk: {st.session_state.disk_threshold}%
            """
        )

        if st.button("Reset to Defaults", use_container_width=True):
            st.session_state.cpu_threshold = 80
            st.session_state.memory_threshold = 85
            st.session_state.disk_threshold = 90
            st.success("Thresholds reset to default values!")
            st.rerun()

    st.markdown("---")
    st.subheader("Appearance")
    dark_mode = st.toggle("Dark Mode", value=st.session_state.dark_mode)
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()

    st.markdown("---")
    st.subheader("Auto-Refresh")
    auto_refresh = st.toggle("Enable Auto-Refresh", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh
    if auto_refresh:
        refresh_interval = st.slider(
            "Refresh Interval (seconds)", 10, 300, st.session_state.refresh_interval, 10
        )
        st.session_state.refresh_interval = refresh_interval
        st.info(f"Dashboard will refresh every {refresh_interval} seconds.")




# --- UI: System Health Check ---
def health_check_page():
    """System Health Check page."""
    st.title("System Health Check")
    st.markdown("---")

    expected_columns = [
        "id",
        "timestamp",
        "cpu",
        "memory",
        "disk",
        "ping_status",
        "ping_ms",
    ]
    required_columns = ["timestamp", "cpu", "memory", "disk"]
    metric_columns = ["cpu", "memory", "disk"]

    try:
        df = get_system_logs()
        if df.empty:
            st.info("No data available yet. Add samples to run the health check.")
            return

        missing_columns = [col for col in expected_columns if col not in df.columns]
        missing_values = 0
        invalid_counts = {"cpu": 0, "memory": 0, "disk": 0}

        for col in required_columns:
            series = df[col]
            missing_mask = series.isna() | series.astype(str).str.strip().eq("")
            missing_values += int(missing_mask.sum())

        for metric in metric_columns:
            series = df[metric]
            missing_mask = series.isna() | series.astype(str).str.strip().eq("")
            numeric = pd.to_numeric(series, errors="coerce")
            invalid_mask = (~missing_mask) & (numeric.isna() | (numeric < 0) | (numeric > 100))
            invalid_counts[metric] = int(invalid_mask.sum())

        ok = (
            not missing_columns
            and missing_values == 0
            and invalid_counts["cpu"] == 0
            and invalid_counts["memory"] == 0
            and invalid_counts["disk"] == 0
        )

        status_message = "All checks passed." if ok else "Issues detected."
        if ok:
            st.success(status_message)
        else:
            st.warning(status_message)

        st.subheader("Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", len(df))
        c2.metric("Missing Values", missing_values)
        c3.metric("Invalid CPU", invalid_counts["cpu"])
        c4.metric("Invalid Memory", invalid_counts["memory"])

        c5, c6, c7 = st.columns(3)
        c5.metric("Invalid Disk", invalid_counts["disk"])
        c6.metric("Missing Columns", len(missing_columns))
        c7.metric("Range Check", "PASS" if ok else "FAIL")

        if missing_columns:
            st.error(f"Missing columns: {', '.join(missing_columns)}")

        st.markdown("---")
        st.subheader("Latest Sample")
        latest = df.head(1)
        st.dataframe(latest, use_container_width=True, hide_index=True)

    except Exception as exc:
        st.error(f"Health check failed: {exc}")

# --- UI: About ---
def about_page():
    """About page."""
    st.title("About")
    st.markdown("---")
    st.markdown(
        """
        ## System Monitor Dashboard

        Version: 3.0.0  
        Features: Live data collection, authentication, configurable thresholds.

        The dashboard stores samples in SQLite (log.db) and tracks:
        - CPU, Memory, Disk usage
        - Network ping status and response time
        - Alerts when thresholds are exceeded

        Default credentials: admin / admin123
        """
    )


# --- UI: Sidebar & Navigation ---
def sidebar_navigation():
    """Create sidebar navigation menu."""
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align: center; padding: 16px 0;'>
                <h1 style='margin: 0; font-size: 22px;'>SysMonitor</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.logged_in:
            st.success(f"Logged in as: {st.session_state.username} ({st.session_state.role})")

        st.markdown("---")
        st.header("Navigation")
        pages = ["Dashboard", "System Health Check", "About"]
        if st.session_state.role == "admin":
            pages.insert(1, "Configuration")
        page = st.radio("Select Page", pages, label_visibility="collapsed")
        st.markdown("---")

        st.header("Quick Actions")
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        if st.button("Logout", use_container_width=True, type="primary"):
            logout()

        st.markdown("---")
        st.header("Appearance")
        dark_toggle = st.toggle("Dark Mode", value=st.session_state.dark_mode)
        if dark_toggle != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_toggle
            st.rerun()

        if st.session_state.auto_refresh:
            st.success(f"Auto-refresh: {st.session_state.refresh_interval}s")
        else:
            st.info("Auto-refresh: Disabled")

        st.markdown("---")
        st.header("Current Thresholds")
        st.metric("CPU", f"{st.session_state.cpu_threshold}%")
        st.metric("Memory", f"{st.session_state.memory_threshold}%")
        st.metric("Disk", f"{st.session_state.disk_threshold}%")

        st.markdown("---")
        st.header("Database")
        try:
            stats = get_statistics(
                (
                    st.session_state.cpu_threshold,
                    st.session_state.memory_threshold,
                    st.session_state.disk_threshold,
                )
            )
            st.metric("Total Records", stats["log_count"])
            st.metric("Total Alerts", stats["alert_count"])
            st.metric("Violations", stats["threshold_violations"])
        except Exception:
            st.warning("Database not connected")

        st.caption("System Monitor v3.0")

    return page


# --- App Entrypoint ---
def main():
    """Main application entry point."""
    init_db()

    thresholds = {
        "cpu": st.session_state.cpu_threshold,
        "memory": st.session_state.memory_threshold,
        "disk": st.session_state.disk_threshold,
    }
    sample_interval = (
        max(MIN_SAMPLE_SECONDS, st.session_state.refresh_interval)
        if st.session_state.auto_refresh
        else max(MIN_SAMPLE_SECONDS, 10)
    )
    new_sample, new_alerts = collect_metrics(thresholds, sample_interval)

    if not st.session_state.logged_in:
        login_page()
        return

    page = sidebar_navigation()
    if page == "Dashboard":
        dashboard_page(new_alerts)
    elif page == "System Health Check":
        health_check_page()
    elif page == "Configuration" and st.session_state.role == "admin":
        configuration_page()
    elif page == "Configuration":
        st.error("Access denied: admin role required.")
    elif page == "About":
        about_page()

    if st.session_state.auto_refresh and page == "Dashboard":
        time.sleep(st.session_state.refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
