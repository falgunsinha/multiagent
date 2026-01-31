import json

# Read the notebook
with open('multiagent_test_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell with plot_bar_with_error and replace it
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'plot_bar_with_error' in ''.join(cell.get('source', [])):
        # Replace with only plot_metric_by_episode
        cell['source'] = [
            'def plot_metric_by_episode(metrics_data, metric_key, ylabel, color_map, title):\n',
            '    """Plot metric across episodes (average across both seeds, no error bars)"""\n',
            '    \n',
            '    with sns.axes_style("whitegrid", rc={\'font.family\': plt.rcParams[\'font.family\']}):\n',
            '        fig, ax = plt.subplots(figsize=(10, 6))\n',
            '        \n',
            '        for model, data in metrics_data.items():\n',
            '            episodes = data[\'episodes\']\n',
            '            values = data[metric_key]\n',
            '            \n',
            '            # Skip if all values are None (e.g., Heuristic reward)\n',
            '            if all(v is None for v in values):\n',
            '                continue\n',
            '            \n',
            '            color = color_map.get(model, \'#000000\')\n',
            '            ax.plot(episodes, values, marker=\'o\', label=model, color=color, \n',
            '                   linewidth=2, markersize=6)\n',
            '        \n',
            '        ax.set_xlabel(\'Episodes\', fontsize=12)\n',
            '        ax.set_ylabel(ylabel, fontsize=12)\n',
            '        ax.set_title(title, fontsize=12, pad=15)\n',
            '        \n',
            '        # Set x-axis ticks to match target episodes\n',
            '        ax.set_xticks([2, 4, 6, 8, 10, 12, 14, 16, 18])\n',
            '        \n',
            '        ax.legend(loc=\'best\', frameon=True, fontsize=10)\n',
            '        \n',
            '        ax.spines[\'right\'].set_visible(False)\n',
            '        ax.spines[\'top\'].set_visible(False)\n',
            '        ax.grid(True, alpha=0.3, linestyle=\'--\', linewidth=0.5)\n',
            '        \n',
            '        plt.tight_layout()\n',
            '        plt.show()\n',
            '\n',
            'print("✅ Plotting functions defined")'
        ]
        print(f'✅ Found and updated cell {i}')
        break

# Write back
with open('multiagent_test_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('✅ Successfully updated the notebook')

