import psutil
from datetime import datetime
import sqlite3
import os
import time
import subprocess
import platform

DB_NAME = "log.db"

# Alert Thresholds
CPU_THRESHOLD = 80.0
MEMORY_THRESHOLD = 85.0
DISK_THRESHOLD = 90.0

def init_db():
    """Initialize database with system_log and alerts_log tables"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Create system_log table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            cpu REAL,
            memory REAL,
            disk REAL,
            ping_status TEXT,
            ping_ms REAL
        )
    """)
    
    # Bonus: Create alerts_log table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            alert_type TEXT,
            value REAL,
            threshold REAL,
            message TEXT
        )
    """)
    
    conn.commit()
    conn.close()

def get_system_info():
    """Collect current system metrics"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cpu = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory().percent
    disk = psutil.disk_usage('/').percent
    ping_status, ping_ms = ping_host("8.8.8.8")
    return (now, cpu, memory, disk, ping_status, ping_ms)

def ping_host(host):
    """Ping a host and return status and response time"""
    try:
        param = "-n" if platform.system().lower() == "windows" else "-c"
        output = subprocess.check_output(
            ["ping", param, "1", host], 
            stderr=subprocess.DEVNULL
        ).decode()
        ms = parse_ping_time(output)
        return ("UP", ms)
    except:
        return ("DOWN", -1)

def parse_ping_time(output):
    """Extract ping time from ping command output"""
    for line in output.splitlines():
        if "time=" in line.lower() or "time<" in line.lower():
            if "time=" in line.lower():
                parts = line.lower().split("time=")
                if len(parts) > 1:
                    try:
                        time_str = parts[1].split()[0].replace("ms", "").strip()
                        return float(time_str)
                    except (ValueError, IndexError):
                        pass
            elif "time<" in line.lower():
                parts = line.lower().split("time<")
                if len(parts) > 1:
                    try:
                        time_str = parts[1].split()[0].replace("ms", "").strip()
                        return float(time_str)
                    except (ValueError, IndexError):
                        pass
    return -1

def insert_log(data):
    """Insert one row of system info into SQLite"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO system_log (timestamp, cpu, memory, disk, ping_status, ping_ms)
        VALUES (?, ?, ?, ?, ?, ?)
    """, data)
    conn.commit()
    conn.close()

def insert_alert(timestamp, alert_type, value, threshold, message):
    """Insert alert record into alerts_log table"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO alerts_log (timestamp, alert_type, value, threshold, message)
        VALUES (?, ?, ?, ?, ?)
    """, (timestamp, alert_type, value, threshold, message))
    conn.commit()
    conn.close()

def check_alerts(data):
    """Check if any system metrics exceed thresholds and trigger alerts"""
    timestamp, cpu, memory, disk, ping_status, ping_ms = data
    alerts_triggered = []
    
    # Check CPU threshold
    if cpu > CPU_THRESHOLD:
        message = f"⚠️  ALERT: High CPU usage! ({cpu}%)"
        print(message)
        alerts_triggered.append(("CPU", cpu, CPU_THRESHOLD, message))
    
    # Check Memory threshold
    if memory > MEMORY_THRESHOLD:
        message = f"⚠️  ALERT: High Memory usage! ({memory}%)"
        print(message)
        alerts_triggered.append(("MEMORY", memory, MEMORY_THRESHOLD, message))
    
    # Check Disk threshold
    if disk > DISK_THRESHOLD:
        message = f"⚠️  ALERT: Low Disk Space! ({disk}%)"
        print(message)
        alerts_triggered.append(("DISK", disk, DISK_THRESHOLD, message))
    
    # Bonus: Log alerts to database
    for alert_type, value, threshold, message in alerts_triggered:
        insert_alert(timestamp, alert_type, value, threshold, message)
    
    return len(alerts_triggered) > 0

def show_last_entries(limit=5):
    """Retrieve and print the last few records from the database"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM system_log
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    print(f"\n{'='*80}")
    print(f"Last {limit} System Log Entries:")
    print(f"{'='*80}")
    for row in reversed(rows):
        print(row)

def show_alerts_log(limit=10):
    """Display recent alerts from alerts_log table"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM alerts_log
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    if rows:
        print(f"\n{'='*80}")
        print(f"Recent Alerts (Last {limit}):")
        print(f"{'='*80}")
        for row in reversed(rows):
            print(f"[{row[1]}] {row[2]}: {row[5]}")
    else:
        print(f"\n{'='*80}")
        print("No alerts triggered!")
        print(f"{'='*80}")

def count_total_records():
    """Count and display total records in database"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM system_log")
    log_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM alerts_log")
    alert_count = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"\n{'='*80}")
    print(f"Database Statistics:")
    print(f"  • Total system log records: {log_count}")
    print(f"  • Total alerts recorded: {alert_count}")
    print(f"{'='*80}")

if __name__ == "__main__":
    print("="*80)
    print("WEEK 8: System Monitor with Alert Thresholds")
    print("="*80)
    print(f"Alert Thresholds:")
    print(f"  • CPU: {CPU_THRESHOLD}%")
    print(f"  • Memory: {MEMORY_THRESHOLD}%")
    print(f"  • Disk: {DISK_THRESHOLD}%")
    print("="*80)
    
    # Initialize database
    init_db()
    
    # Insert 5 new log entries (one every 10 seconds)
    print("\nCollecting system metrics (5 samples, 10-second intervals)...\n")
    for i in range(5):
        row = get_system_info()
        insert_log(row)
        print(f"Logged: {row}")
        check_alerts(row)
        
        if i < 4:  # Don't sleep after the last entry
            print()
            time.sleep(10)
    
    # Show last 5 entries
    show_last_entries()
    
    # Show alerts log
    show_alerts_log()
    
    # Count total records
    count_total_records()