import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.image as mpimg
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Load data
df = pd.read_csv("Doomsday_Clock.csv")
years = df["year"].tolist()
minutes = df["min_to_midnight"].tolist()

y_max = 20  # 23:40 on the clock = 20 minutes to midnight

# Build waterfall components
bases = []
increases = []
decreases = []

for i in range(len(minutes)):
    if i == 0:
        bases.append(0)
        increases.append(minutes[i])
        decreases.append(0)
    else:
        change = minutes[i] - minutes[i - 1]
        if change >= 0:
            bases.append(minutes[i - 1])
            increases.append(change)
            decreases.append(0)
        else:
            bases.append(minutes[i])
            increases.append(0)
            decreases.append(abs(change))

x = np.array(years, dtype=float)
bases = np.array(bases)
increases = np.array(increases)
decreases = np.array(decreases)
bar_width = 0.6

fig, ax = plt.subplots(figsize=(20, 10))

# 1947: dashed level line instead of green bar
ax.plot([x[0] - bar_width / 2, x[0] + bar_width / 2],
        [minutes[0], minutes[0]], color="#5F9598", linewidth=1.5,
        linestyle="--", zorder=3)

# Bars for 1949+ (index 1 onward)
ax.bar(x[1:], bases[1:], bar_width, color="none", edgecolor="none")
ax.bar(x[1:], increases[1:], bar_width, bottom=bases[1:], color="#5F9598",
       edgecolor="white", linewidth=0.5)
ax.bar(x[1:], decreases[1:], bar_width, bottom=bases[1:], color="#061E29",
       edgecolor="white", linewidth=0.5)

# Connector lines
for i in range(len(minutes) - 1):
    ax.plot([x[i] + bar_width / 2, x[i + 1] - bar_width / 2],
            [minutes[i], minutes[i]], color="dimgray", linewidth=0.8,
            linestyle="--", zorder=3)

# Y-axis
ax.set_ylim(0, y_max)
ax.set_yticks([0, 5, 10, 15, y_max])
ax.set_yticklabels(["00:00", "23:55", "23:50", "23:45", "23:40"])
ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))

# X-axis: ticks for every year 1947-2026, labels only at data years
all_years = list(range(1947, 2027))
ax.set_xticks(all_years)
min_gap = 3
xlabels = []
last_shown = -999
for y in all_years:
    if y in years and y - last_shown >= min_gap:
        xlabels.append(str(y))
        last_shown = y
    else:
        xlabels.append("")
ax.set_xticklabels(xlabels, rotation=0, ha="center", fontsize=8)
ax.set_xlim(1945, 2028)

ax.set_xlabel("")
ax.set_ylabel("")

# Spines: only bottom and left
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(True)
ax.spines["left"].set_visible(True)

# Ticks
ax.tick_params(axis="x", length=4, direction="out")
ax.tick_params(axis="y", length=0)

# Y-axis title at the top of the axis
ax.text(-0.01, 1.01, "Time until midnight", transform=ax.transAxes,
        fontsize=10, ha="left", va="bottom", fontweight="bold", color="#061E29")

# ── Event landmarks with actual historical dates ─────────────────────
# (actual_date_fractional_year, data_year, label)
events = [
    (1945.60, 1949, "Nukes dropped"),          # Aug 6 1945 — Hiroshima
    (1952.84, 1953, "The Super is tested"),     # Nov 1 1952 — Ivy Mike H-bomb
    (1962.79, 1963, "Cuban crisis"),            # Oct 16 1962
    (1968.08, 1968, "Vietnam"),                 # Jan 30 1968 — Tet Offensive
    (1969.88, 1969, "SALT"),                    # Nov 17 1969 — SALT talks begin
    (1974.38, 1974, "India"),                   # May 18 1974 — Smiling Buddha
    (1979.98, 1980, "Afghanistan"),             # Dec 24 1979 — Soviet invasion
    (1986.32, 1988, "Chernobyl"),              # Apr 26 1986
    (1991.98, 1991, "Collapse of USSR"),        # Dec 26 1991
    (1998.36, 1998, "South Asia tests"),        # May 11 1998
    (2001.69, 2002, "9/11"),                    # Sep 11 2001
    (2006.77, 2007, "DPRK"),                    # Oct 9 2006 — first nuclear test
    (2009.95, 2010, "2\u00b0C"),               # Dec 2009 — Copenhagen summit
    (2011.19, 2012, "Chaos"),                   # Mar 11 2011 — Fukushima
    (2017.05, 2017, "Trump"),                   # Jan 20 2017 — inauguration
    (2018.07, 2018, "Cybersecurity"),           # Jan 25 2018 — clock announcement
    (2022.15, 2023, "Ukraine"),                 # Feb 24 2022 — Russian invasion
    (2023.77, 2023, "Oct 7"),                   # Oct 7 2023 — Hamas attack
    (2024.50, 2025, "AI"),                      # mid-2024 — AI concerns peak
    (2025.05, 2025, "Trump"),                   # Jan 20 2025 — 2nd inauguration
    (2026.08, 2026, "Chaos"),                   # Jan 2026 — clock setting
]

# Get bar top for a data year (top of the waterfall bar)
def get_bar_top(yr):
    idx = years.index(yr)
    if idx == 0:
        return minutes[0]
    return max(minutes[idx], minutes[idx - 1])

# Group events that share the same actual date position (within 0.5 yr)
# by combining their labels
grouped_events = []
for actual, data_yr, label in events:
    merged = False
    for i, (a, d, lbls) in enumerate(grouped_events):
        if abs(a - actual) < 0.4:
            grouped_events[i] = (a, d, lbls + " / " + label)
            merged = True
            break
    if not merged:
        grouped_events.append((actual, data_yr, label))

# Greedy label placement
CHAR_W = 0.35
LINE_H = 0.65
X_PAD = 0.5
Y_PAD = 0.4
LABEL_FONTSIZE = 10

def box_overlaps(b, placed):
    x1, x2, y1, y2 = b
    for px1, px2, py1, py2 in placed:
        if x1 < px2 and x2 > px1 and y1 < py2 and y2 > py1:
            return True
    return False

placed_boxes = []
label_specs = []

# Also reserve a box for the pronouncement text area so labels avoid it
# Pronouncement will be around x=2005-2027, y=14-17
placed_boxes.append((2002, 2028, 10.5, 17.5))

for actual, data_yr, text in sorted(grouped_events, key=lambda e: e[0]):
    bar_top = get_bar_top(data_yr)
    text_w = len(text) * CHAR_W
    text_h = LINE_H

    # Try placing label at increasing heights above bar top
    for offset in np.arange(Y_PAD, 19, 0.3):
        label_y = bar_top + offset
        box = (actual + X_PAD, actual + X_PAD + text_w,
               label_y - 0.1, label_y + text_h)
        if not box_overlaps(box, placed_boxes):
            break

    placed_boxes.append(box)
    label_specs.append((actual, bar_top, label_y, text))

# Draw vertical lines (full height to bottom) and labels
for actual, bar_top, label_y, text in label_specs:
    # Gray line from bottom (0) up to label position
    ax.plot([actual, actual], [0, label_y + LINE_H * 0.5],
            color="#061E29", linewidth=0.5, alpha=0.25, zorder=1)
    # Label to the right of the line
    ax.text(actual + X_PAD, label_y, text, ha="left", va="bottom",
            fontsize=LABEL_FONTSIZE, color="#061E29", alpha=0.85)

# ── Logo (top-right corner of the chart) ─────────────────────────────
try:
    logo = mpimg.imread("logo.png")
    imagebox = OffsetImage(logo, zoom=0.055)
    ab = AnnotationBbox(imagebox, (1.0, 1.0), xycoords="axes fraction",
                        box_alignment=(1.0, 1.0), frameon=False,
                        pad=0.2, zorder=5)
    ax.add_artist(ab)
except Exception:
    pass

# "85 seconds to midnight" below logo
ax.text(1.0, 0.93, "85 seconds to midnight",
        transform=ax.transAxes, fontsize=9, ha="right", va="top",
        color="#061E29", fontstyle="italic")

# ── Pronouncement (large, lower, clearly visible) ────────────────────
ax.text(2015, 13, "The world is closer to\ndestruction than ever.",
        fontsize=26, ha="center", va="top", fontstyle="italic",
        color="#061E29", fontweight="bold", zorder=5)

plt.tight_layout()
plt.savefig("Doomsday_Clock.png", dpi=200, bbox_inches="tight")
plt.show()
