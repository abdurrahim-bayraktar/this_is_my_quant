import sys
sys.path.insert(0, ".")
from dashboard.backend.data_loader import scan_reports

exps = scan_reports()
print(f"Found {len(exps)} experiments\n")
for e in exps[:15]:
    sent = "SENT" if e["use_sentiment"] else "no-sent"
    bt = "BT" if e["has_backtest"] else "--"
    folds = f"F{e['n_folds']}" if e["has_folds"] else "--"
    print(f"  {e['name']:50s} {e['task']:15s} {sent:7s} {bt:3s} {folds:4s} {e['headline_metric_name']}={e['headline_metric_value']:.4f}")
