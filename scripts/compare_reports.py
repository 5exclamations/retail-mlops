"""CI helper: two training reports from identical inputs must agree on metrics (reproducibility check)."""
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
ok = a["metrics"] == b["metrics"] and a["selected_model"] == b["selected_model"]
print("reproducible" if ok else f"MISMATCH\n{a['metrics']}\n{b['metrics']}")
sys.exit(0 if ok else 1)
