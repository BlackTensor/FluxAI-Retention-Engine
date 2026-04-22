"""Chaos Monkey Tests for FluxAI Data Validation"""
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_utils import validate_upload

results = []

def test(name, passed):
    status = "PASS" if passed else "FAIL"
    results.append((name, status))
    print(f"  [{status}] {name}")

print("=" * 60)
print("FLUXAI CHAOS MONKEY TESTS")
print("=" * 60)

# TEST 1: Empty file
print("\nTEST 1: Empty file (0 rows)")
df = pd.DataFrame()
valid, cleaned, errors, warnings = validate_upload(df)
test("Empty file rejected", not valid)
test("Error message present", len(errors) > 0)

# TEST 2: None input
print("\nTEST 2: None input")
valid, cleaned, errors, warnings = validate_upload(None)
test("None input rejected", not valid)

# TEST 3: Wrong headers with fuzzy matching
print("\nTEST 3: Fuzzy column matching")
df = pd.DataFrame({
    'Tenure_Months': [24, 48],
    'monthly_charges': [79.85, 56.30],
    'total_charges': [1889.50, 2680.20],
    'contract_type': ['Month-to-month', 'Two year'],
})
valid, cleaned, errors, warnings = validate_upload(df)
test("Fuzzy columns accepted", valid)
if valid and cleaned is not None:
    test("tenure column exists after rename", 'tenure' in cleaned.columns)
    test("MonthlyCharges column exists after rename", 'MonthlyCharges' in cleaned.columns)
    test("Contract column exists after rename", 'Contract' in cleaned.columns)
    test("Rename warnings generated", len(warnings) > 0)
    print(f"    Warnings: {warnings}")

# TEST 4: Data type nightmare
print("\nTEST 4: Non-numeric text in number columns")
df = pd.DataFrame({
    'tenure': [24, 'N/A', 10],
    'MonthlyCharges': [79.85, 'Free', 'Premium'],
    'TotalCharges': [1889.50, '', 'Included'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
})
valid, cleaned, errors, warnings = validate_upload(df)
test("Bad data types accepted (coerced)", valid)
if valid and cleaned is not None:
    test("MonthlyCharges are numeric", all(isinstance(v, (int, float)) for v in cleaned['MonthlyCharges']))
    test("TotalCharges are numeric", all(isinstance(v, (int, float)) for v in cleaned['TotalCharges']))
    test("tenure is numeric", all(isinstance(v, (int, float)) for v in cleaned['tenure']))
    test("Non-numeric warnings generated", any('non-numeric' in w for w in warnings))
    print(f"    MonthlyCharges: {cleaned['MonthlyCharges'].tolist()}")
    print(f"    TotalCharges: {cleaned['TotalCharges'].tolist()}")
    print(f"    tenure: {cleaned['tenure'].tolist()}")

# TEST 5: Completely wrong columns
print("\nTEST 5: Completely wrong columns")
df = pd.DataFrame({'name': ['Alice'], 'age': [30], 'city': ['NYC']})
valid, cleaned, errors, warnings = validate_upload(df)
test("Wrong columns rejected", not valid)
test("Missing column error present", any('Missing' in e for e in errors))

# TEST 6: Single row of valid data
print("\nTEST 6: Minimal valid data (1 row)")
df = pd.DataFrame({
    'tenure': [12],
    'MonthlyCharges': [50.0],
    'TotalCharges': [600.0],
    'Contract': ['Month-to-month'],
})
valid, cleaned, errors, warnings = validate_upload(df)
test("Minimal valid data accepted", valid)
test("Default columns filled in", 'customerID' in cleaned.columns if cleaned is not None else False)

# SUMMARY
print("\n" + "=" * 60)
passed = sum(1 for _, s in results if s == "PASS")
failed = sum(1 for _, s in results if s == "FAIL")
print(f"RESULTS: {passed}/{len(results)} passed, {failed} failed")
if failed > 0:
    print("FAILED TESTS:")
    for name, status in results:
        if status == "FAIL":
            print(f"  - {name}")
else:
    print("ALL TESTS PASSED!")
print("=" * 60)
