
import os
import pandas as pd
import plotly.express as px

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================

df = pd.read_csv("data/geo_distribution.csv")

df.columns = df.columns.str.strip()

for col in ["year", "val", "lower", "upper"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Keep prevalence percentage records only
percent = df[
    (df["measure_name"] == "Prevalence") &
    (df["metric_name"] == "Percent") &
    (df["cause_name"] == "Attention-deficit/hyperactivity disorder")
].copy()

# GBD percentages are stored as fractions
percent["prevalence"] = percent["val"] * 100
percent["lower_percent"] = percent["lower"] * 100
percent["upper_percent"] = percent["upper"] * 100

# Create output folder
os.makedirs("charts", exist_ok=True)

# ==========================================
# 2. SETTINGS
# ==========================================

selected_year = 2023
selected_age = "5-9 years"
selected_sex = "Female"
selected_country = "Kyrgyzstan"

# ==========================================
# 3. AGE DISTRIBUTION
# ==========================================

age_data = percent[
    percent["year"] == selected_year
].copy()

age_order = [
    "2-4 years", "5-9 years", "10-14 years",
    "15-19 years", "20-24 years", "25-29 years",
    "30-34 years", "35-39 years", "40-44 years",
    "45-49 years", "50-54 years", "55-59 years",
    "60-64 years", "65-69 years", "70-74 years",
    "75-79 years", "80+ years"
]

fig1 = px.bar(
    age_data,
    x="age_name",
    y="prevalence",
    color="sex_name",
    barmode="group",
    category_orders={"age_name": age_order},
    error_y=age_data["upper_percent"] - age_data["prevalence"],
    error_y_minus=age_data["prevalence"] - age_data["lower_percent"],
    title="ADHD Prevalence by Age and Sex — 2023",
    labels={
        "age_name": "Age group",
        "prevalence": "Prevalence (%)",
        "sex_name": "Sex"
    }
)

fig1.update_layout(
    xaxis_tickangle=-45,
    template="plotly_white"
)

fig1.write_html("charts/01_age_distribution.html")

# ==========================================
# 4. COUNTRY COMPARISON
# ==========================================

country_data = percent[
    (percent["year"] == selected_year) &
    (percent["age_name"] == selected_age)
].copy()

fig2 = px.bar(
    country_data,
    x="location_name",
    y="prevalence",
    color="sex_name",
    barmode="group",
    error_y=country_data["upper_percent"] - country_data["prevalence"],
    error_y_minus=country_data["prevalence"] - country_data["lower_percent"],
    title=f"ADHD Prevalence by Country — {selected_age}, {selected_year}",
    labels={
        "location_name": "Country",
        "prevalence": "Prevalence (%)",
        "sex_name": "Sex"
    }
)

fig2.update_layout(
    xaxis_tickangle=-55,
    template="plotly_white"
)

fig2.write_html("charts/02_country_comparison.html")

# ==========================================
# 5. COUNTRY MAP
# ==========================================

map_data = percent[
    (percent["year"] == selected_year) &
    (percent["age_name"] == selected_age) &
    (percent["sex_name"] == selected_sex)
].copy()

# Geographic aliases for map recognition
map_data["map_country"] = map_data["location_name"].replace({
    "Republic of Korea": "South Korea",
    "Russian Federation": "Russia",
    "United States of America": "United States",
    "Viet Nam": "Vietnam"
})

fig3 = px.choropleth(
    map_data,
    locations="map_country",
    locationmode="country names",
    color="prevalence",
    hover_name="location_name",
    hover_data={
        "prevalence": ":.2f",
        "lower_percent": ":.2f",
        "upper_percent": ":.2f",
        "map_country": False
    },
    color_continuous_scale="YlOrRd",
    title=f"ADHD Prevalence Map — {selected_sex}, {selected_age}, {selected_year}",
    labels={"prevalence": "Prevalence (%)"}
)

fig3.update_geos(
    showcoastlines=True,
    showcountries=True,
    projection_type="natural earth"
)

fig3.write_html("charts/03_geographic_map.html")

# ==========================================
# 6. TIME TREND
# ==========================================

trend_data = percent[
    (percent["location_name"] == selected_country) &
    (percent["age_name"] == selected_age)
].copy()

fig4 = px.line(
    trend_data,
    x="year",
    y="prevalence",
    color="sex_name",
    markers=True,
    error_y=trend_data["upper_percent"] - trend_data["prevalence"],
    error_y_minus=trend_data["prevalence"] - trend_data["lower_percent"],
    title=f"ADHD Prevalence Trend — {selected_country}, {selected_age}",
    labels={
        "year": "Year",
        "prevalence": "Prevalence (%)",
        "sex_name": "Sex"
    }
)

fig4.update_layout(
    template="plotly_white",
    xaxis=dict(tickmode="linear", dtick=1)
)

fig4.write_html("charts/04_time_trend.html")

# ==========================================
# 7. FINISHED
# ==========================================

print("\nSUCCESS! Your visualizations are ready.")

print("\nFiles saved in the charts folder:")
for file in os.listdir("charts"):
    print(" -", file)

print("\nDataset summary:")
print("Countries:", percent["location_name"].nunique())
print("Years:", sorted(percent["year"].unique()))
print("Age groups:", percent["age_name"].nunique())
print("Sex categories:", sorted(percent["sex_name"].unique()))
