# =============================================================================
# 🦕 PANDAS TUTORIAL: JURASSIC DATA PARK 🦖
# =============================================================================
# Welcome! In this tutorial, you'll learn the Pandas library by exploring
# real fossil records of dinosaurs. We assume you know Python basics —
# variables, loops, functions — but are new to Pandas.
#
# Dataset columns:
#   occurence_no  – original fossil occurrence ID
#   name          – dinosaur genus name
#   diet          – omnivorous / carnivorous / herbivorous
#   type          – small theropod, large theropod, sauropod, etc.
#   length_m      – max body length (meters)
#   max_ma        – earliest fossil record age (million years ago)
#   min_ma        – latest fossil record age (million years ago)
#   region        – where the fossil was found
#   lng / lat     – coordinates of the fossil site
#   class         – Saurischia or Ornithischia
#   family        – taxonomic family (if known)
#
# To run this script you need:
#   pip install pandas matplotlib
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Quick display helper — prints a titled separator before any output
def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# =============================================================================
# SECTION 1 — LOADING & INSPECTING DATA
# =============================================================================
# pd.read_csv() is your entry point into almost every Pandas workflow.
# It reads a CSV file and returns a DataFrame — the core Pandas data structure,
# basically a table with labelled rows and columns.
# =============================================================================

section("1. LOADING & INSPECTING DATA")

df = pd.read_csv("dinosaurs.csv")

# --- Shape & Size ---
print(f"\n📐 Shape (rows, columns): {df.shape}")
print(f"   Total cells: {df.size}")

# --- Peek at the data ---
print("\n🔍 First 5 rows — df.head()")
print(df.head())

print("\n🔍 Last 5 rows — df.tail()")
print(df.tail())

# --- Column names & dtypes ---
print("\n📋 Columns:", df.columns.tolist())

print("\n🔢 Data types per column — df.dtypes")
print(df.dtypes)
# Pandas uses:
#   object  → string / mixed
#   float64 → decimal numbers
#   int64   → whole numbers

# --- Quick statistical overview ---
print("\n📊 Numerical summary — df.describe()")
print(df.describe())
# describe() covers count, mean, std, min, quartiles, max

print("\n📊 All columns (including text) — df.describe(include='all')")
print(df.describe(include="all"))
# include='all' also shows unique counts and top values for strings

# --- Column-by-column info ---
print("\n🗂️  DataFrame info — df.info()")
df.info()
# info() shows non-null counts — great for spotting missing data at a glance


# =============================================================================
# SECTION 2 — HANDLING MISSING VALUES
# =============================================================================
# Real-world data is messy. Pandas gives you clear tools to find, fill, or
# drop missing values (represented as NaN — "Not a Number").
# =============================================================================

section("2. HANDLING MISSING VALUES")

# --- Detect missing values ---
print("\n❓ Missing values per column — df.isna().sum()")
print(df.isna().sum())

# isna() returns a boolean DataFrame (True where value is missing)
# .sum() then counts the Trues per column

missing_pct = (df.isna().sum() / len(df) * 100).round(2)
print("\n📉 Missing % per column:")
print(missing_pct[missing_pct > 0])  # only show columns that have gaps

# --- Rows with ANY missing value ---
rows_with_na = df[df.isna().any(axis=1)]
print(f"\n🚨 Rows with at least one missing value: {len(rows_with_na)}")

# --- Strategy 1: Drop rows with missing values ---
df_dropped = df.dropna()
print(f"\n🗑️  After dropna(): {df_dropped.shape[0]} rows remain "
      f"(dropped {len(df) - len(df_dropped)})")

# --- Strategy 2: Fill missing values ---
# Fill missing numeric length with the column median
median_length = df["length_m"].median()
df["length_m"] = df["length_m"].fillna(median_length)
print(f"\n🩹 Filled missing length_m with median ({median_length:.2f} m)")

# Fill missing family with a placeholder string
df["family"] = df["family"].fillna("Unknown Family")
print("🩹 Filled missing family with 'Unknown Family'")

# Confirm no more missing in those columns
print(f"\n✅ Missing in length_m after fill: {df['length_m'].isna().sum()}")
print(f"✅ Missing in family after fill:    {df['family'].isna().sum()}")


# =============================================================================
# SECTION 3 — FILTERING & QUERYING
# =============================================================================
# Filtering lets you zoom in on the rows you care about.
# Pandas supports two styles: bracket notation and .query()
# =============================================================================

section("3. FILTERING & QUERYING")

# --- Single condition ---
carnivores = df[df["diet"] == "carnivorous"]
print(f"\n🥩 Carnivores: {len(carnivores)} dinosaurs")

# --- Multiple conditions (use & for AND, | for OR, wrap each in parentheses) ---
big_carnivores = df[(df["diet"] == "carnivorous") & (df["length_m"] > 10)]
print(f"🦷 Carnivores longer than 10 m: {len(big_carnivores)}")
print(big_carnivores[["name", "length_m", "region"]].head())

# --- .isin() — match any of a list ---
north_america_asia = df[df["region"].isin(["North America", "Asia"])]
print(f"\n🌏 Fossils from North America or Asia: {len(north_america_asia)}")

# --- .query() — SQL-like syntax, often more readable ---
ancient_giants = df.query("max_ma > 150 and length_m > 20")
print(f"\n🏛️  Dinosaurs older than 150 Ma AND longer than 20 m: {len(ancient_giants)}")
print(ancient_giants[["name", "max_ma", "length_m"]].head())

# --- String filtering with .str accessor ---
sauro_family = df[df["family"].str.contains("sauridae", case=False, na=False)]
print(f"\n🔤 Families containing 'sauridae': {len(sauro_family)} records")

# --- Negation filter (~ means NOT) ---
non_herbivores = df[~(df["diet"] == "herbivorous")]
print(f"\n🚫 Non-herbivores: {len(non_herbivores)}")


# =============================================================================
# SECTION 4 — SORTING & RANKING
# =============================================================================
# Sorting helps surface extremes — the biggest, oldest, most common, etc.
# =============================================================================

section("4. SORTING & RANKING")

# --- Sort by a single column ---
by_length = df.sort_values("length_m", ascending=False)
print("\n📏 Top 10 longest dinosaurs:")
print(by_length[["name", "length_m", "type"]].head(10).to_string(index=False))

# --- Sort by multiple columns ---
by_diet_length = df.sort_values(["diet", "length_m"], ascending=[True, False])
print("\n🍽️  Sorted by diet (A→Z) then length (longest first):")
print(by_diet_length[["name", "diet", "length_m"]].head(8).to_string(index=False))

# --- .nlargest() / .nsmallest() — shortcut for top/bottom N ---
print("\n🏆 5 most recently recorded dinosaurs (smallest min_ma):")
print(df.nsmallest(5, "min_ma")[["name", "min_ma", "region"]].to_string(index=False))

print("\n🕰️  5 oldest dinosaurs (largest max_ma):")
print(df.nlargest(5, "max_ma")[["name", "max_ma", "region"]].to_string(index=False))

# --- .rank() — add a rank column ---
df["length_rank"] = df["length_m"].rank(ascending=False, method="min").astype(int)
print("\n🥇 Dinosaurs with their length rank (top 5):")
print(df.nsmallest(5, "length_rank")[["name", "length_m", "length_rank"]].to_string(index=False))


# =============================================================================
# SECTION 5 — GROUPBY & AGGREGATIONS
# =============================================================================
# GroupBy splits the data into groups, applies a function to each group,
# and combines the results. This is the "split → apply → combine" pattern.
# =============================================================================

section("5. GROUPBY & AGGREGATIONS")

# --- Count by group ---
diet_counts = df.groupby("diet").size().sort_values(ascending=False)
print("\n🍽️  Dinosaur count by diet:")
print(diet_counts.to_string())

# --- Mean of a numeric column per group ---
avg_length_by_type = df.groupby("type")["length_m"].mean().sort_values(ascending=False)
print("\n📐 Average length (m) by dinosaur type:")
print(avg_length_by_type.round(2).to_string())

# --- Multiple aggregations at once with .agg() ---
size_stats = df.groupby("type")["length_m"].agg(
    count="count",
    mean="mean",
    median="median",
    max="max"
).round(2).sort_values("mean", ascending=False)
print("\n📊 Length statistics by type:")
print(size_stats.to_string())

# --- Multi-column groupby ---
diet_class = df.groupby(["diet", "class"]).agg(
    count=("name", "count"),
    avg_length=("length_m", "mean")
).round(2)
print("\n🦴 Count & avg length by diet + taxonomic class:")
print(diet_class.to_string())

# --- Regional diversity: unique dinosaur types per region ---
region_diversity = df.groupby("region")["type"].nunique().sort_values(ascending=False)
print("\n🌍 Unique dinosaur types per region:")
print(region_diversity.to_string())

# --- transform() — adds group result back to the original DataFrame ---
df["avg_length_in_type"] = df.groupby("type")["length_m"].transform("mean")
print("\n🔁 transform() adds group mean back to each row (first 5 rows):")
print(df[["name", "type", "length_m", "avg_length_in_type"]].head().to_string(index=False))


# =============================================================================
# SECTION 6 — MERGING & RESHAPING
# =============================================================================
# Real projects often have multiple tables. Pandas merge() works just like
# SQL JOINs. Reshaping (pivot, melt) lets you change the table's structure.
# =============================================================================

section("6. MERGING & RESHAPING")

# We'll create a small supplementary table to demonstrate merging
era_map = pd.DataFrame({
    "era_label": ["Triassic", "Jurassic", "Cretaceous"],
    "era_start": [251, 201, 145],
    "era_end":   [201, 145,  66]
})
print("\n📖 Era reference table:")
print(era_map.to_string(index=False))

# Assign era based on max_ma (when the dino first appeared)
def assign_era(max_ma):
    if max_ma > 201:
        return "Triassic"
    elif max_ma > 145:
        return "Jurassic"
    else:
        return "Cretaceous"

df["era_label"] = df["max_ma"].apply(assign_era)

# Now merge — adds era_start / era_end columns to our main DataFrame
df_merged = df.merge(era_map, on="era_label", how="left")
print(f"\n🔗 After merge: {df_merged.shape} (gained {df_merged.shape[1] - df.shape[1]} columns)")
print(df_merged[["name", "max_ma", "era_label", "era_start", "era_end"]].head().to_string(index=False))

# --- pivot_table() — like Excel pivot tables ---
pivot = df.pivot_table(
    values="length_m",
    index="era_label",
    columns="diet",
    aggfunc="mean"
).round(2)
print("\n📊 Pivot table — mean length by era & diet:")
print(pivot.to_string())

# --- melt() — the opposite of pivot: wide → long format ---
# Let's melt max_ma and min_ma into a single "time" column
time_cols = df[["name", "max_ma", "min_ma"]].head(6)
melted = time_cols.melt(id_vars="name", var_name="time_type", value_name="million_years_ago")
print("\n🔄 melt() — unpivot max_ma & min_ma into long format:")
print(melted.to_string(index=False))

# --- value_counts() for quick frequency tables ---
print("\n📈 Top 10 most represented regions:")
print(df["region"].value_counts().head(10).to_string())


# =============================================================================
# SECTION 7 — PLOTTING WITH PANDAS
# =============================================================================
# Pandas wraps Matplotlib to give you quick, one-line plots directly from
# DataFrames and Series. Perfect for exploratory data analysis.
# =============================================================================

section("7. PLOTTING WITH PANDAS")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("🦕 Jurassic Data Park — Pandas Plots", fontsize=18, fontweight="bold", y=1.01)

# ── Plot 1: Bar chart — count by diet ──────────────────────────────────────
diet_counts.plot(
    kind="bar", ax=axes[0, 0],
    color=["#e07b54", "#5b8db8", "#6abf69"],
    edgecolor="white", width=0.6
)
axes[0, 0].set_title("Dinosaurs by Diet")
axes[0, 0].set_xlabel("Diet")
axes[0, 0].set_ylabel("Count")
axes[0, 0].tick_params(axis="x", rotation=30)

# ── Plot 2: Horizontal bar — average length by type ────────────────────────
avg_length_by_type.plot(
    kind="barh", ax=axes[0, 1],
    color="#7b68ee", edgecolor="white"
)
axes[0, 1].set_title("Average Length by Dinosaur Type")
axes[0, 1].set_xlabel("Length (m)")
axes[0, 1].invert_yaxis()   # largest on top

# ── Plot 3: Histogram — distribution of body length ────────────────────────
df["length_m"].plot(
    kind="hist", bins=30, ax=axes[0, 2],
    color="#f4a261", edgecolor="white"
)
axes[0, 2].set_title("Distribution of Body Length")
axes[0, 2].set_xlabel("Length (m)")
axes[0, 2].set_ylabel("Frequency")

# ── Plot 4: Pie chart — class breakdown ────────────────────────────────────
df["class"].value_counts().plot(
    kind="pie", ax=axes[1, 0],
    autopct="%1.1f%%", startangle=90,
    colors=["#e9c46a", "#264653"],
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
axes[1, 0].set_title("Saurischia vs Ornithischia")
axes[1, 0].set_ylabel("")

# ── Plot 5: Scatter — length vs age (max_ma) ───────────────────────────────
diet_colors = {"carnivorous": "#e63946", "herbivorous": "#2a9d8f", "omnivorous": "#e9c46a"}
for diet_type, group in df.groupby("diet"):
    group.plot.scatter(
        x="max_ma", y="length_m",
        ax=axes[1, 1],
        color=diet_colors.get(diet_type, "grey"),
        alpha=0.5, s=20,
        label=diet_type
    )
axes[1, 1].set_title("Body Length vs Age")
axes[1, 1].set_xlabel("First Appearance (Million Years Ago)")
axes[1, 1].set_ylabel("Max Length (m)")
axes[1, 1].legend(title="Diet")
axes[1, 1].invert_xaxis()   # time flows right → left (older on left)

# ── Plot 6: Stacked bar — era × diet ───────────────────────────────────────
era_diet = df.groupby(["era_label", "diet"]).size().unstack(fill_value=0)
era_diet = era_diet.reindex(["Triassic", "Jurassic", "Cretaceous"])
era_diet.plot(
    kind="bar", stacked=True, ax=axes[1, 2],
    color=["#e63946", "#6abf69", "#e9c46a"],
    edgecolor="white", width=0.5
)
axes[1, 2].set_title("Dinosaurs per Era by Diet")
axes[1, 2].set_xlabel("Era")
axes[1, 2].set_ylabel("Count")
axes[1, 2].tick_params(axis="x", rotation=30)
axes[1, 2].legend(title="Diet")

plt.tight_layout()
plt.savefig("dino_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n💾 Plot saved as dino_plots.png")


# =============================================================================
# SECTION 8 — BONUS: PUTTING IT ALL TOGETHER
# =============================================================================
# A mini-analysis pipeline combining everything you've learned.
# Question: "Which regions produced the most diverse & largest dinosaurs?"
# =============================================================================

section("8. BONUS — MINI ANALYSIS PIPELINE")

print("\n🔬 Question: Which regions had the largest & most diverse dinosaurs?\n")

region_report = (
    df.groupby("region")
    .agg(
        total_fossils   = ("name",     "count"),
        unique_dinos    = ("name",     "nunique"),
        unique_types    = ("type",     "nunique"),
        avg_length_m    = ("length_m", "mean"),
        max_length_m    = ("length_m", "max"),
        pct_carnivore   = ("diet",     lambda x: (x == "carnivorous").mean() * 100)
    )
    .round(2)
    .sort_values("avg_length_m", ascending=False)
)

print(region_report.to_string())

# Top 3 by average size
print("\n🏆 Top 3 regions by average dinosaur size:")
print(region_report.nlargest(3, "avg_length_m")[["avg_length_m", "unique_dinos", "pct_carnivore"]].to_string())

print("\n✅ Tutorial complete! You've covered:")
print("   1. Loading & Inspecting  |  2. Missing Values")
print("   3. Filtering & Querying  |  4. Sorting & Ranking")
print("   5. GroupBy & Aggregation |  6. Merging & Reshaping")
print("   7. Plotting              |  8. Mini Pipeline")
print("\n🦕 Happy wrangling!\n")