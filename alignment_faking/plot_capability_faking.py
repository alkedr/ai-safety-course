import os
import json
import subprocess
import matplotlib.pyplot as plt
from collections import defaultdict

# Define the logs directory
logs_dir = "alignment_faking/logs"

# Store data as: plot_data[(model_name, base_scenario_name)] = {"free_score": float, "paid_score": float}
plot_data = defaultdict(dict)

# Regex to extract task function name from filenames like "task_gpqa_free_consequences-anthropic_claude_blah.eval"
# Not used if header_data["task_id"] is reliable
# import re
# filename_pattern = re.compile(r"task_(gpqa_[a-zA-Z0-9_]+(?:free|paid)[a-zA-Z0-9_]*)-.*\.eval")

for model_dir in os.listdir(logs_dir):
    model_path = os.path.join(logs_dir, model_dir)
    if not os.path.isdir(model_path) or model_dir.lower() in ["old", "r1", ".ipynb_checkpoints"]:
        continue

    model_name_from_dir = model_dir

    eval_files = [
        f for f in os.listdir(model_path)
        if "gpqa" in f.lower() and f.endswith(".eval") # Check for 'gpqa' anywhere in the name (case-insensitive)
    ]
    # if not eval_files: # If no gpqa files in this model_dir, skip to next model_dir
    #     continue

    # # --- Start: Debug block to print header structure for the first file and exit ---
    # first_eval_file_path = os.path.join(model_path, eval_files[0])
    # print(f"DEBUG: Inspecting header structure of: {first_eval_file_path}")
    # try:
    #     debug_header_json = subprocess.check_output(
    #         ["inspect", "log", "dump", "--header-only", first_eval_file_path],
    #         text=True,
    #         stderr=subprocess.PIPE
    #     )
    #     debug_header_data = json.loads(debug_header_json)
    #     print("DEBUG: 'reductions' field content:")
    #     print(json.dumps(debug_header_data.get("reductions"), indent=2))
    #     print("DEBUG: 'results.scores' field content:")
    #     print(json.dumps(debug_header_data.get("results", {}).get("scores"), indent=2))
    # except Exception as e:
    #     print(f"DEBUG: Error during header inspection: {e}")
    # exit() # Exit after inspecting the first file
    # # --- End: Debug block ---

    for eval_file in eval_files:
        eval_path = os.path.join(model_path, eval_file)
        try:
            header_json = subprocess.check_output(
                ["inspect", "log", "dump", "--header-only", eval_path],
                text=True,
                stderr=subprocess.PIPE
            )
            header_data = json.loads(header_json)
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"Error processing {eval_path}: {e}")
            continue

        actual_model_name = (list(header_data.get("stats", {}).get("model_usage", {}).keys()) or [model_name_from_dir])[0]

        task_function_name = header_data.get("task", {}).get("function")

        if not task_function_name:
            parts = eval_file.split('_')
            if len(parts) > 1: # Check if there are underscores to split by
                potential_task_names = []
                # Try to find a segment that looks like a task name
                # e.g., gpqa-free-consequences from 2025-05-08T23-47-28+03-00_gpqa-free-consequences_LPNLCZDBzN7jJEmidu46Mz.eval
                # The task name part often contains "gpqa"
                for part in parts:
                    if "gpqa" in part.lower() and ("free" in part.lower() or "paid" in part.lower()):
                        potential_task_names.append(part.replace("-", "_"))
                if potential_task_names:
                    task_function_name = sorted(potential_task_names, key=len, reverse=True)[0]
                    # print(f"Debug: No task.function found for {eval_path}. Inferred '{task_function_name}' from filename part.")
                else:
                    # If specific pattern not found, try a more general approach if filename has at least 3 parts typical of logs
                    if len(parts) > 2 and "gpqa" in parts[1].lower(): # Heuristic: TIMESTAMP_TASKNAME_HASH
                         task_function_name = parts[1].replace("-","_")
                         # print(f"Debug: No task.function found for {eval_path}. Inferred '{task_function_name}' from filename part (heuristic).")

        if not task_function_name:
            print(f"Warning: Could not determine task function name for {eval_path}. Skipping.")
            continue

        # Path based on new understanding from debug output
        accuracy = header_data.get("results", {}).get("scores", [{}])[0].get("metrics", {}).get("accuracy", {}).get("value")

        # Fallback to reductions if the above is None (less likely now, but for safety)
        if accuracy is None:
            # print(f"DEBUG: Accuracy not found in results.scores[0].metrics.accuracy.value for {eval_file}. Trying reductions[0].value.") # Removed this debug line
            accuracy = header_data.get("reductions", [{}])[0].get("value")

        # Fallback to the old results.scores[0].value if still None (even less likely, but for extra safety)
        if accuracy is None:
            # print(f"DEBUG: Accuracy not found in reductions for {eval_file}. Trying results.scores[0].value as a last resort.") # Removing this debug line too
            accuracy = header_data.get("results", {}).get("scores", [{}])[0].get("value")

        # print(f"DEBUG: File: {eval_file}, TaskFunc: {task_function_name}, RawAccuracy: {accuracy}") # DEBUG PRINT

        is_free = False
        is_paid = False
        base_scenario_name = task_function_name # Default if no free/paid found

        # Standardize separators for parsing (e.g. from filename inference)
        parsed_task_name = task_function_name.replace("-", "_")

        free_indicator = "_free"
        paid_indicator = "_paid"

        current_indicator_details = None
        if free_indicator in parsed_task_name:
            current_indicator_details = (free_indicator, "free")
        elif paid_indicator in parsed_task_name: # Use elif to ensure only one is chosen
            current_indicator_details = (paid_indicator, "paid")

        if current_indicator_details:
            indicator_str, status_type = current_indicator_details
            if status_type == "free":
                is_free = True
            else: # paid
                is_paid = True

            parts = parsed_task_name.split(indicator_str, 1)
            prefix = parts[0]
            suffix_from_split = parts[1] if len(parts) > 1 else ""

            if suffix_from_split:
                base_scenario_name = f"{prefix.rstrip('_')}{suffix_from_split}"
            else:
                base_scenario_name = prefix.rstrip('_')

            if parsed_task_name.startswith("gpqa") and not base_scenario_name.startswith("gpqa"):
                 base_scenario_name_temp_no_gpqa = base_scenario_name.replace("gpqa_", "")
                 base_scenario_name = "gpqa_" + base_scenario_name_temp_no_gpqa
        else:
            print(f"Warning: Task function name '{task_function_name}' from {eval_path} "
                  f"does not contain '{free_indicator}' or '{paid_indicator}'. Using full name as base scenario and cannot determine free/paid status.")

        # Clean up base_scenario_name: remove any remaining gpqa_ prefix if already present, and ensure it starts with gpqa
        base_scenario_name = base_scenario_name.replace("gpqa_gpqa_", "gpqa_") # In case of duplication
        if base_scenario_name.startswith("gpqa_"):
            base_scenario_name = base_scenario_name[5:] # Remove "gpqa_"
        # remove _refuse suffix if present
        if base_scenario_name.endswith("_refuse"):
            base_scenario_name = base_scenario_name[:-7]


        current_key = (actual_model_name, base_scenario_name)
        if is_free:
            plot_data[current_key]['free_score'] = accuracy
        elif is_paid:
            plot_data[current_key]['paid_score'] = accuracy


# DEBUG PRINT for plot_data before sorting keys
print("DEBUG: Final plot_data before creating vals_for_plot:")
# Convert tuple keys to string for json.dumps to work
printable_plot_data = {str(key): value for key, value in plot_data.items()}
print(json.dumps(printable_plot_data, indent=2))

# Sort by model name (case-insensitive), then base_scenario_name (case-insensitive)
# Using reverse=True to potentially match ordering of example plot (e.g. newer/more prominent models at top)
sorted_keys = sorted(plot_data.keys(), key=lambda x: (x[0].lower(), x[1].lower()), reverse=True)

y_tick_labels = [f"{model} ({scenario})" for model, scenario in sorted_keys]
vals_for_plot = []

for key in sorted_keys:
    data_point = plot_data[key]
    free_score = data_point.get('free_score')
    paid_score = data_point.get('paid_score')

    # If score is None (e.g. task failed or 0 samples), treat accuracy as 0.
    free_accepted = free_score if free_score is not None else 0.0
    free_rejected = 1.0 - free_accepted # Simplified

    paid_accepted = paid_score if paid_score is not None else 0.0
    paid_rejected = 1.0 - paid_accepted # Simplified

    vals_for_plot.append({
        "free_accepted": free_accepted, "free_rejected": free_rejected, "free_score_raw": free_score,
        "paid_accepted": paid_accepted, "paid_rejected": paid_rejected, "paid_score_raw": paid_score,
    })

if not vals_for_plot:
    print("No data to plot. Exiting.")
    exit()

plt.figure(figsize=(14, max(7, len(sorted_keys) * 0.8))) # Increased width slightly, dynamic height
width = 0.35       # Width of each individual bar (free or paid)
y_pos = range(len(sorted_keys)) # y-positions for each (model, scenario) group
bar_offset = width / 2 # Offset for free vs paid bars from the center y_pos

# Plot bars for free and paid
plot_items = [
    ("free", -bar_offset, "free_accepted", "free_rejected"),
    ("paid",  bar_offset, "paid_accepted", "paid_rejected"),
]

for i, (category_label, y_cat_offset, acc_key, rej_key) in enumerate(plot_items):
    # y-coordinates for the current category's bars (e.g., all "free" bars)
    current_y_values = [y + y_cat_offset for y in y_pos]

    accepted_segment_widths = [v[acc_key] for v in vals_for_plot]
    rejected_segment_widths = [v[rej_key] for v in vals_for_plot]

    # Plot 'accepted' segment (green)
    plt.barh(current_y_values, accepted_segment_widths, width, color="green",
             label='correct' if i == 0 else '_nolegend_')

    # Plot 'rejected' segment (red)
    # 'left' is now directly accepted_segment_widths
    plt.barh(current_y_values, rejected_segment_widths, width, color='red',
             label='incorrect' if i == 0 else '_nolegend_',
             left=accepted_segment_widths)

plt.yticks(y_pos, y_tick_labels, fontsize=9)
plt.legend(loc='lower right', bbox_to_anchor=(1, 0)) # Adjusted legend position
plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95]) # Add some padding

# Add "free" and "paid" text labels next to the bars
label_x = -0.04 # Use a fixed x-position for labels

# Ensure x-axis starts from a point that makes labels visible
plt.xlim(-0.12, 1.05) # Use fixed x-axis limits


for idx, y_val in enumerate(y_pos):
    # Check if data exists for free/paid to avoid plotting text if no bar exists
    has_free_data = vals_for_plot[idx]['free_score_raw'] is not None
    has_paid_data = vals_for_plot[idx]['paid_score_raw'] is not None

    if has_free_data:
      plt.text(label_x, y_val - bar_offset, 'free', va='center', ha='right', fontsize=9, color='black', fontweight='bold')
    if has_paid_data:
      plt.text(label_x, y_val + bar_offset, 'paid', va='center', ha='right', fontsize=9, color='black', fontweight='bold')


output_filename = 'capability_faking.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")

# To show the plot interactively (optional, comment out if not needed for automated runs)
# plt.show()
