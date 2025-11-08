# Week 14 – Loops & Arrays: Complete Testing of All Features

## Objective

This week, you will test **all components of your monitoring system** to ensure correctness and completeness.  
You’ll use Python loops and validation logic to check your database, data quality, and metric ranges.

---

## Tasks

1. Verify that your `log.db` file exists and is readable.
2. Check your `system_log` table for:
   - Missing columns
   - Empty or invalid data
3. Validate numeric ranges for CPU, Memory, and Disk (0–100)
4. Print a summary report of your testing results.
5. (Optional) Save the test summary to a text file.

---

## Example Output

🔍 Running Full System Test...

✅ Database file found.

✅ Loaded 120 records from system_log.

✅ Column check passed.

✅ No missing values detected.

✅ All system metrics within valid range (0–100).

===== Test Summary =====

Total Records: 120

Missing Values: 0

Invalid CPU Records: 0

🟢 System validation complete.


---

## Submission Checklist

- [ ] `test_script.py` runs without error  
- [ ] Summary output printed correctly  
- [ ] Screenshot of terminal test results  
- [ ] Code pushed to GitHub

---

## Bonus (Optional)

- Save the test results into `test_report.txt`
- Add automated re-run if any invalid records are found
- Create a small dashboard page “System Health Check”

