def pie_dominant(rgb_colors, proportions, labels=None):
    import matplotlib.pyplot as plt
    import numpy as np

    if labels is None:
        labels = [f"{round(p * 100, 1)}%" for p in proportions]

    plt.figure(figsize=(5, 5))
    plt.pie(
        proportions,
        labels=labels,
        colors=[np.array(c) / 255 for c in rgb_colors],
        startangle=90
    )
    plt.title("Dominant Hue Pie Chart 🎨")
    plt.axis("equal")
    plt.show()
