import argparse
import sqlite3
import time
from pathlib import Path

DB_NAME = "log.db"
EXPECTED_COLUMNS = [
    "id",
    "timestamp",
    "cpu",
    "memory",
    "disk",
    "ping_status",
    "ping_ms",
]
REQUIRED_COLUMNS = ["timestamp", "cpu", "memory", "disk"]
METRIC_COLUMNS = ["cpu", "memory", "disk"]


def is_number(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value.strip())
            return True
        except ValueError:
            return False
    return False


def to_float(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        return float(value.strip())
    raise ValueError("Value is not numeric")


def build_summary_lines(stats):
    status_line = "\nSystem validation complete." if stats["ok"] else "System validation found issues."
    lines = [
        "===== Test Summary =====",
        f"\nTotal Records: {stats['total_records']}",
        f"\nMissing Values: {stats['missing_values']}",
        f"\nInvalid CPU Records: {stats['invalid_cpu']}",
        f"\nInvalid Memory Records: {stats['invalid_memory']}",
        f"\nInvalid Disk Records: {stats['invalid_disk']}",
        status_line,
    ]
    return lines


def run_checks():
    db_path = Path(DB_NAME)
    if not db_path.exists():
        print("FAIL: Database file not found.")
        return {"fatal": True, "ok": False}

    try:
        conn = sqlite3.connect(DB_NAME)
    except sqlite3.Error as exc:
        print(f"FAIL: Database file not readable ({exc}).")
        return {"fatal": True, "ok": False}

    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='system_log'"
        )
        table_row = cursor.fetchone()
        if not table_row:
            print("FAIL: system_log table not found.")
            return {"fatal": True, "ok": False}

        cursor.execute("PRAGMA table_info(system_log)")
        columns = [row[1] for row in cursor.fetchall()]
        missing_columns = [col for col in EXPECTED_COLUMNS if col not in columns]

        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM system_log")
        row_objects = cursor.fetchall()
        total_records = len(row_objects)

        missing_values = 0
        invalid_cpu = 0
        invalid_memory = 0
        invalid_disk = 0

        for row in row_objects:
            for col in REQUIRED_COLUMNS:
                value = row[col]
                if value is None or (isinstance(value, str) and not value.strip()):
                    missing_values += 1

            for metric in METRIC_COLUMNS:
                value = row[metric]
                if value is None or (isinstance(value, str) and not value.strip()):
                    continue
                if not is_number(value):
                    if metric == "cpu":
                        invalid_cpu += 1
                    elif metric == "memory":
                        invalid_memory += 1
                    else:
                        invalid_disk += 1
                    continue

                metric_value = to_float(value)
                if metric_value < 0 or metric_value > 100:
                    if metric == "cpu":
                        invalid_cpu += 1
                    elif metric == "memory":
                        invalid_memory += 1
                    else:
                        invalid_disk += 1

        print("OK: Database file found.")
        print(f"OK: Loaded {total_records} records from system_log.")
        if missing_columns:
            print(f"FAIL: Missing columns: {', '.join(missing_columns)}")
        else:
            print("OK: Column check passed.")

        if missing_values == 0:
            print("OK: No missing values detected.")
        else:
            print(f"WARN: Missing values detected: {missing_values}")

        if invalid_cpu == 0 and invalid_memory == 0 and invalid_disk == 0:
            print("OK: All system metrics within valid range (0-100).")
        else:
            print(
                "WARN: Invalid metric values found. "
                f"CPU={invalid_cpu}, Memory={invalid_memory}, Disk={invalid_disk}"
            )

        ok = (
            not missing_columns
            and missing_values == 0
            and invalid_cpu == 0
            and invalid_memory == 0
            and invalid_disk == 0
        )

        return {
            "fatal": False,
            "ok": ok,
            "total_records": total_records,
            "missing_values": missing_values,
            "invalid_cpu": invalid_cpu,
            "invalid_memory": invalid_memory,
            "invalid_disk": invalid_disk,
            "missing_columns": missing_columns,
        }
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Run full system_log validation checks.")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the test summary to test_report.txt",
    )
    parser.add_argument(
        "--report",
        default="test_report.txt",
        help="Path for saving the test summary (used with --save)",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Re-run the test automatically if invalid records are found.",
    )
    parser.add_argument(
        "--rerun-attempts",
        type=int,
        default=2,
        help="How many times to re-run when invalid records are found.",
    )
    parser.add_argument(
        "--rerun-delay",
        type=int,
        default=5,
        help="Seconds to wait between re-runs.",
    )
    args = parser.parse_args()

    print("Running Full System Test...")

    stats = run_checks()
    if stats.get("fatal"):
        return

    attempts_left = max(0, args.rerun_attempts)
    while args.rerun and not stats["ok"] and attempts_left > 0:
        attempts_left -= 1
        print(f"Re-running test in {args.rerun_delay} seconds... ({attempts_left} retries left)")
        time.sleep(max(0, args.rerun_delay))
        stats = run_checks()
        if stats.get("fatal"):
            return

    summary_lines = build_summary_lines(stats)
    print("" + "".join(summary_lines))

    if args.save:
        report_path = Path(args.report)
        report_path.write_text("".join(summary_lines) + "", encoding="utf-8")
        print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
