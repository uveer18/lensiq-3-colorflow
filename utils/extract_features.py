def extract_color_features(image_path, k=3):
    import cv2
    import numpy as np
    from sklearn.cluster import KMeans
    from collections import Counter
    from itertools import combinations

    # Load and resize
    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 512))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    # Averages
    avg_hue = np.mean(pixels[:, 0])
    avg_sat = np.mean(pixels[:, 1])
    avg_val = np.mean(pixels[:, 2])

    # Colorfulness
    B, G, R = cv2.split(img.astype("float"))
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    colorfulness = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)

    # Warmth
    warm_pixels = np.sum((pixels[:, 0] < 30) | ((pixels[:, 0] > 150) & (pixels[:, 0] < 180)))
    cool_pixels = np.sum((pixels[:, 0] > 90) & (pixels[:, 0] < 150))
    warmth_score = 1.0 if cool_pixels == 0 else warm_pixels / (cool_pixels + 1e-6)

    # KMeans for dominant colors (HSV & BGR)
    km = KMeans(n_clusters=k, random_state=42).fit(pixels)
    labels = km.labels_
    total_pixels = len(labels)
    counts = Counter(labels)
    proportions = sorted([counts[i] / total_pixels for i in range(k)], reverse=True)
    dominant_hues = km.cluster_centers_[:, 0] / 180.0
    dominant_rgb = []

    for hsv_val in km.cluster_centers_:
        hsv_pixel = np.uint8([[hsv_val]])
        bgr = cv2.cvtColor(hsv_pixel.astype("uint8"), cv2.COLOR_HSV2BGR)[0][0]
        rgb = tuple(int(c) for c in bgr[::-1])
        dominant_rgb.append(rgb)

    # Harmony
    def hue_distance(h1, h2):
        abs1 = abs(h1 - h2)
        return min(abs1, 180 - abs1)

    def harmony_score(dominant_hues):
        pairs = list(combinations(dominant_hues, 2))
        distances = [hue_distance(h1 * 180, h2 * 180) for h1, h2 in pairs]
        harmony1 = sum(
            1 if any(abs(d - h) < 15 for h in [30, 60, 120, 180]) else 0 for d in distances
        )
        return harmony1 / len(pairs) if len(pairs)!=0 else harmony1

    harmony = harmony_score(dominant_hues)

    # Feature vector (scaled)
    feature_vector = [
        avg_hue / 180,
        avg_sat / 255,
        avg_val / 255,
        warmth_score,
        colorfulness / 100,
        *list(dominant_hues),
        harmony,
    ]

    return feature_vector, dominant_rgb, proportions
