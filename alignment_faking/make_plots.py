import os
import json
import subprocess
import matplotlib.pyplot as plt

# Define the logs directory
logs_dir = "alignment_faking/logs"

model_data = {}

for model_dir in os.listdir(logs_dir):
    model_path = os.path.join(logs_dir, model_dir)
    if not os.path.isdir(model_path) or model_dir == "old":
        continue
    eval_files = [f for f in os.listdir(model_path) if f.endswith('.eval') and 'alignment-faking' in f]
    for eval_file in eval_files:
        eval_path = os.path.join(model_path, eval_file)
        try:
            header_json = subprocess.check_output([
                "inspect", "log", "dump", "--header-only", eval_path
            ], text=True)
            header_data = json.loads(header_json)
        except Exception:
            continue
        model_name = (list(header_data.get("stats", {}).get("model_usage", {}).keys()) or [model_dir])[0]
        metrics = header_data.get("results", {}).get("scores", [{}])[0].get("metrics", {})
        vals = [metrics.get(k, {}).get("value") for k in ["free_accepted", "free_rejected", "paid_accepted", "paid_rejected"]]
        if all(v is not None for v in vals):
            model_data[(model_name, eval_file)] = dict(zip(["free_accepted", "free_rejected", "paid_accepted", "paid_rejected"], vals))

models = sorted(model_data.keys(), key=lambda x: (x[0], x[1]), reverse=True)
vals = [model_data[m] for m in models]

plt.figure(figsize=(12, max(6, len(models)*0.7)))
width = 0.35
y_pos = range(len(models))
bar_offset = width / 2

# Plot bars for free and paid
for i, (label, offset, acc, rej) in enumerate([
    ("free", -bar_offset, "free_accepted", "free_rejected"),
    ("paid",  bar_offset, "paid_accepted", "paid_rejected"),
]):
    plt.barh([y + offset for y in y_pos], [v[acc] for v in vals], width, color="red", label='accepted' if i == 0 else '_nolegend_')
    plt.barh([y + offset for y in y_pos], [v[rej] for v in vals], width, color='green', label='rejected' if i == 0 else '_nolegend_', left=[1.0 - v[rej] for v in vals])

plt.yticks(y_pos, [m[0] for m in models])
plt.xlabel('Metric Value')
plt.legend(loc='lower right')
plt.tight_layout()

label_x = -0.04
for y in y_pos:
    plt.text(label_x, y - bar_offset, 'free', va='center', ha='right', fontsize=9, color='black', fontweight='bold')
    plt.text(label_x, y + bar_offset, 'paid', va='center', ha='right', fontsize=9, color='black', fontweight='bold')

plt.xlim(-0.12, 1.05)
plt.savefig('alignment_faking.png')
print("Plot saved as alignment_faking.png")
