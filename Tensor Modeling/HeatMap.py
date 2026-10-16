import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. Define the models and the extracted metrics
models = ["nnUNet", "SimpleUNet", "GATSegDiff"]
metrics = [
    "Total Patches Tested",
    "Det=0 Count"
]

# 2. Assemble the populated data matrix (Rows = Models, Columns = Metrics)
# Your exact counts from the logs are mapped directly here
data = np.array([
    [550, 166],  # nnUNet metrics
    [60, 25],    # SimpleUNet metrics
    [270, 63]    # GATSegDiff metrics
])

# 3. Create the heatmap plot
plt.figure(figsize=(10, 6))

# Annotate with integer formatting, use a clear sequential colormap, and set edge colors
sns.heatmap(
    data,
    annot=True,
    fmt="d",
    cmap="YlOrRd",
    xticklabels=metrics,
    yticklabels=models,
    linewidths=1.5,
    linecolor="white",
    cbar_kws={"label": "Patch / Space Count"},
)

# 4. Add clear labels and title
plt.title(
    "Dynamic Subspace Accuracy & Structural Stability Report",
    fontsize=14,
    fontweight="bold",
    pad=20,
)
plt.xlabel("Dimentionality Ratio", fontsize=12, labelpad=10)
plt.ylabel("Model Architectures", fontsize=12, labelpad=10)

# Adjust layout to make sure labels don't get cut off
plt.tight_layout()

# 5. Display the plot
plt.show()


