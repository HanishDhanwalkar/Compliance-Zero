import re

RISK_PATTERNS = {
    "sanctions": r"\bsanctions?\b",
    "aml": r"\bAML\b",
    "kyc_failure": r"\bKYC\b.*\bfailure\b"
}

def run_risk_rules(text: str):
    flags = []
    for name, pattern in RISK_PATTERNS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            flags.append(name)
    return flags
