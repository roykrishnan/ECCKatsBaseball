import os
import random
from datetime import datetime

# Hardcode the target date
TARGET_DATE = datetime(2026, 2, 15)


def random_time_str():
    """Return a random time string between 9:00:00 AM and 3:00:00 PM, e.g. '2:34:17 PM'"""
    total_seconds = random.randint(9 * 3600, 15 * 3600)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    period = "AM" if h < 12 else "PM"
    display_h = h if h <= 12 else h - 12
    return f"{display_h}:{m:02d}:{s:02d} {period}"


def format_target_date(time_str):
    """Build the target date string: Sun Feb 15 2026 2:34:17 PM"""
    day_name = TARGET_DATE.strftime("%a")
    month    = TARGET_DATE.strftime("%b")
    day      = TARGET_DATE.day
    year     = TARGET_DATE.year
    return f"{day_name} {month} {day} {year} {time_str}"


def process_csv(filepath):
    filename = os.path.basename(filepath)

    with open(filepath, "r", newline="", encoding="utf-8-sig") as f:
        raw = f.read()

    lines = raw.splitlines(keepends=True)
    delimiter = "\t" if "\t" in lines[0] else ","

    # Find header row containing "Date"
    header_row_idx = None
    date_col_idx = None

    for i, line in enumerate(lines):
        cols = line.rstrip("\n").split(delimiter)
        if "Date" in cols:
            header_row_idx = i
            date_col_idx = cols.index("Date")
            break

    if header_row_idx is None:
        print(f"  Skipping {filename} — no 'Date' column found.")
        return

    new_lines = []
    for i, line in enumerate(lines):
        if i <= header_row_idx:
            new_lines.append(line)
            continue

        cols = line.rstrip("\n").split(delimiter)
        if len(cols) <= date_col_idx or not cols[date_col_idx].strip():
            new_lines.append(line)
            continue

        cols[date_col_idx] = format_target_date(random_time_str())
        new_lines.append(delimiter.join(cols) + "\n")

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"  Updated: {filename}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    csv_files = [
        os.path.join(script_dir, f)
        for f in os.listdir(script_dir)
        if f.lower().endswith(".csv")
    ]

    if not csv_files:
        print("No CSV files found in the script directory.")
        return

    print(f"Found {len(csv_files)} CSV file(s) to process...\n")
    for filepath in csv_files:
        process_csv(filepath)

    print(f"\nDone. All dates set to Sun Feb 15 2026.")


if __name__ == "__main__":
    main()