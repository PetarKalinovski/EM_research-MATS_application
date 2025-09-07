import matplotlib.pyplot as plt

# Data for the models
models = ["Qwen2.5\n(Risky Financial Advice)", "Qwen3-4B\n(Risky Financial Advice)"]
em_response_percent = [43, 77.7]  # Replace with your actual data

# Create the bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(
    models,
    em_response_percent,
    color=["#2E86AB", "#A23B72"],
    alpha=0.8,
    edgecolor="black",
    linewidth=1,
)

# Customize the chart
ax.set_ylabel("Percent EM Responses", fontsize=12, fontweight="bold")
ax.set_xlabel("Models", fontsize=12, fontweight="bold")
ax.set_title(
    "Comparison of EM Response Rates for Risky Financial Advice",
    fontsize=14,
    fontweight="bold",
    pad=20,
)

# Set y-axis to go from 0 to 100 (assuming percentages)
ax.set_ylim(0, 100)
ax.set_ylabel("Percent EM Responses (%)")

# Add grid for better readability
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

# Add value labels on top of bars
for bar, value in zip(bars, em_response_percent):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 1,
        f"{value}%",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

# Improve layout
plt.tight_layout()

# Display the chart
plt.show()


plt.savefig("model_comparison_em_responses.png", dpi=300, bbox_inches="tight")
