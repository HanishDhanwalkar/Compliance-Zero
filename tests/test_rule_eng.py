from src.risk_rule_engine import run_risk_rules

results = []
# 1. Sanctions detection
results.append(run_risk_rules("This company is under sanctions by OFAC."))
assert  results[-1] == ["sanctions"]

# 2. AML detection (case-insensitive)
results.append(run_risk_rules("The bank violated aml procedures."))
assert  results[-1] == ["aml"]

# 3. KYC failure detection (pattern requires KYC ... failure)
results.append(run_risk_rules("Customer had a KYC process failure last month."))
assert  results[-1] == ["kyc_failure"]

print("All tests passed!")
print("results:\n", results)
