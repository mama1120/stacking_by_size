import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('stacking_results.csv')

# Filter rows where stacking was successful
success_df = df[df['Stacking_Success']]

# Calculate the counts for correct and incorrect grips for stacking successes
success_counts = success_df.groupby(['Cube_Index', 'cube_gripped_successfully']).size().unstack(fill_value=0)

# Ensure both columns ('Correct Grip' and 'Incorrect Grip') exist
success_counts = success_counts.rename(columns={True: 'Correct Grip', False: 'Incorrect Grip'})
if 'Correct Grip' not in success_counts:
    success_counts['Correct Grip'] = 0
if 'Incorrect Grip' not in success_counts:
    success_counts['Incorrect Grip'] = 0

# Compute total stacking successes for each cube
success_counts['Total Success'] = success_counts.sum(axis=1)

# Get the maximum number of iterations
max_iterations = df['Iteration'].max()

# Plot the stacked bar chart
ax = success_counts[['Correct Grip', 'Incorrect Grip']].plot(
    kind='bar', stacked=True, figsize=(10, 6), color=['green', 'orange']
)

# Add total stacking successes as labels above the bars
for idx, total in enumerate(success_counts['Total Success']):
    ax.text(idx, total + 0.5, str(total), ha='center', va='bottom', fontsize=10, color='black')

# Add titles and labels
plt.title(f'Stacking Success, after {max_iterations} iterations', fontsize=14)
plt.xlabel('Cube Index', fontsize=12)
plt.ylabel('Number of Stacking Successes', fontsize=12)
plt.legend(title='Grip Status', fontsize=10)
plt.xticks(rotation=0)
plt.tight_layout()
plt.grid(True, axis='y')

# Display the plotpick_orientation
plt.show()
