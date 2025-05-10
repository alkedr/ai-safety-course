import os
import json
import subprocess
import matplotlib.pyplot as plt

# Define the logs directory
logs_dir = "alignment_faking/logs"

model_data = {}

for model_dir in os.listdir(logs_dir):
    model_path = os.path.join(logs_dir, model_dir)
    if not os.path.isdir(model_path) or model_dir == "old" or model_dir == "r1":
        continue
    eval_files = [f for f in os.listdir(model_path) if f.endswith('.eval') and 'alignment-faking' in f]
    for eval_file in eval_files:
        eval_path = os.path.join(model_path, eval_file)
        header_json = subprocess.check_output([
            "inspect", "log", "dump", "--header-only", eval_path
        ], text=True)
        header_data = json.loads(header_json)
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
for i, (category_label, y_offset, acc_key, rej_key) in enumerate([
    ("free", -bar_offset, "free_accepted", "free_rejected"),
    ("paid",  bar_offset, "paid_accepted", "paid_rejected"),
]):
    current_y_values = [y + y_offset for y in y_pos] # y-coordinates for the current group (free/paid)

    # Widths for each segment of the bar
    accepted_segment_widths = [v[acc_key] for v in vals]
    rejected_segment_widths = [v[rej_key] for v in vals]

    # Calculate gap widths, ensuring they are not negative
    # This represents the portion of the bar not covered by 'accepted' or 'rejected'
    gap_segment_widths = [max(0, 1.0 - v[acc_key] - v[rej_key]) for v in vals]

    # Plot 'accepted' segment (red)
    # Starts at 0 (implicit left=0)
    plt.barh(current_y_values, accepted_segment_widths, width, color="red",
             label='accepted' if i == 0 else '_nolegend_')

    # Plot 'gap' segment (gray)
    # Starts after the 'accepted' segment
    # The 'left' positions are the widths of the 'accepted' segments
    plt.barh(current_y_values, gap_segment_widths, width, color="gray",
             label='no <OUTPUT> tag (usually rejected)' if i == 0 else '_nolegend_',
             left=accepted_segment_widths)

    # Plot 'rejected' segment (green)
    # Starts after the 'accepted' and 'gap' segments
    # The 'left' positions are the sum of 'accepted' and 'gap' widths
    rejected_segment_lefts = [acc_w + gap_w for acc_w, gap_w in zip(accepted_segment_widths, gap_segment_widths)]
    plt.barh(current_y_values, rejected_segment_widths, width, color='green',
             label='rejected' if i == 0 else '_nolegend_',
             left=rejected_segment_lefts)

plt.yticks(y_pos, [m[0] for m in models])
plt.legend(loc='upper right')
plt.tight_layout()

label_x = -0.04
for y in y_pos:
    plt.text(label_x, y - bar_offset, 'free', va='center', ha='right', fontsize=9, color='black', fontweight='bold')
    plt.text(label_x, y + bar_offset, 'paid', va='center', ha='right', fontsize=9, color='black', fontweight='bold')

plt.xlim(-0.12, 1.05)
plt.savefig('alignment_faking.png')
