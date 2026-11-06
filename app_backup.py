
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from dash import Dash, dcc, html, Input, Output

# =========================================================
# 1. LOAD ALL FIVE CSV FILES
# =========================================================

DATA_FOLDER = Path("data")

FILES = [
    "<1_2019.csv",
    "2019-23.csv",
    "all ages_19-23.csv",
    "geo_dis_<1.csv",
    "geo_distribution.csv"
]

frames = []

for filename in FILES:
    path = DATA_FOLDER / filename

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    temp = pd.read_csv(path)
    temp.columns = temp.columns.str.strip()
    temp["source_file"] = filename
    frames.append(temp)

df = pd.concat(frames, ignore_index=True)

# =========================================================
# 2. CLEAN DATA
# =========================================================

for col in ["year", "val", "lower", "upper"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["year"] = df["year"].astype("Int64")

df = df[
    (df["measure_name"] == "Prevalence") &
    (df["cause_name"] ==
     "Attention-deficit/hyperactivity disorder")
].copy()

# GBD Percent values are fractions.
# Convert to percentage points.
# Number values remain estimated case counts.

df["display_value"] = df["val"]
df["lower_display"] = df["lower"]
df["upper_display"] = df["upper"]

percent_mask = df["metric_name"] == "Percent"

df.loc[percent_mask, "display_value"] *= 100
df.loc[percent_mask, "lower_display"] *= 100
df.loc[percent_mask, "upper_display"] *= 100

# Check duplicate records

KEYS = [
    "location_id",
    "sex_id",
    "age_id",
    "year",
    "metric_id",
    "measure_id",
    "cause_id"
]

if df.duplicated(subset=KEYS).any():
    raise ValueError("Duplicate observations detected!")

# =========================================================
# 3. COUNTRY ISO-3 CODES
# =========================================================

ISO3 = {
    "Australia": "AUS",
    "Austria": "AUT",
    "Brazil": "BRA",
    "Canada": "CAN",
    "China": "CHN",
    "France": "FRA",
    "Germany": "DEU",
    "Ghana": "GHA",
    "Hungary": "HUN",
    "India": "IND",
    "Indonesia": "IDN",
    "Italy": "ITA",
    "Japan": "JPN",
    "Kazakhstan": "KAZ",
    "Kenya": "KEN",
    "Kyrgyzstan": "KGZ",
    "Malaysia": "MYS",
    "Nigeria": "NGA",
    "Republic of Korea": "KOR",
    "Russian Federation": "RUS",
    "Rwanda": "RWA",
    "Spain": "ESP",
    "Thailand": "THA",
    "United Kingdom": "GBR",
    "United States of America": "USA",
    "Uzbekistan": "UZB",
    "Viet Nam": "VNM",
    "Zambia": "ZMB"
}

# =========================================================
# 4. FILTER OPTIONS
# =========================================================

AGE_ORDER = [
    "<1 year",
    "2-4 years",
    "5-9 years",
    "10-14 years",
    "15-19 years",
    "20-24 years",
    "25-29 years",
    "30-34 years",
    "35-39 years",
    "40-44 years",
    "45-49 years",
    "50-54 years",
    "55-59 years",
    "60-64 years",
    "65-69 years",
    "70-74 years",
    "75-79 years",
    "80+ years",
    "All ages"
]

YEARS = sorted(
    df["year"].dropna().astype(int).unique()
)

AGES = [
    age for age in AGE_ORDER
    if age in df["age_name"].dropna().unique()
]

COUNTRIES = sorted(
    df.loc[
        df["location_name"] != "Global",
        "location_name"
    ].dropna().unique()
)

missing_codes = set(COUNTRIES) - set(ISO3)

if missing_codes:
    raise ValueError(
        f"Missing ISO-3 codes: {missing_codes}"
    )

# =========================================================
# 5. DESIGN SETTINGS
# =========================================================

app = Dash(__name__)

app.title = "ADHD Research Dashboard"

BG = "#F4F6FA"
CARD = "#FFFFFF"
NAVY = "#172554"
MUTED = "#64748B"
BORDER = "#E5EAF1"

CARD_STYLE = {
    "backgroundColor": CARD,
    "border": f"1px solid {BORDER}",
    "borderRadius": "12px",
    "padding": "13px",
    "minWidth": "0",
    "boxShadow": "0 2px 6px rgba(15,23,42,0.035)"
}

SECTION_TITLE = {
    "fontSize": "15px",
    "fontWeight": "700",
    "color": NAVY,
    "margin": "0 0 8px 0"
}


# =========================================================
# 6. HELPER FUNCTIONS
# =========================================================

def empty_figure(message):

    fig = go.Figure()

    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14, color=MUTED)
    )

    fig.update_layout(
        template="plotly_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=15, r=15, t=35, b=15)
    )

    return fig


def format_value(value, metric):

    if pd.isna(value):
        return "N/A"

    if metric == "Percent":
        return f"{value:.3f}%"

    return f"{value:,.0f}"


def add_uncertainty(fig, data):

    fig.update_traces(
        error_y=dict(
            type="data",
            symmetric=False,
            array=(
                data["upper_display"] -
                data["display_value"]
            ).clip(lower=0).tolist(),

            arrayminus=(
                data["display_value"] -
                data["lower_display"]
            ).clip(lower=0).tolist()
        )
    )

    return fig


def style_chart(fig, height=350):

    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=35, r=15, t=35, b=40),
        font=dict(
            family="Arial",
            size=11,
            color="#334155"
        ),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    return fig


def normalize_selection(values):

    if values is None:
        return []

    if isinstance(values, str):
        return [values]

    return values


# =========================================================
# 7. DASHBOARD LAYOUT
# =========================================================

app.layout = html.Div(

    style={
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": BG,
        "minHeight": "100vh",
        "padding": "18px 22px"
    },

    children=[

        # HEADER

        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "flexWrap": "wrap",
                "gap": "10px",
                "marginBottom": "14px"
            },

            children=[

                html.Div([

                    html.H2(
                        "ADHD Prevalence Dashboard",
                        style={
                            "margin": "0",
                            "color": NAVY,
                            "fontSize": "24px"
                        }
                    ),

                    html.P(
                        "Global Burden of Disease 2023 | 2019–2023",
                        style={
                            "margin": "5px 0 0",
                            "fontSize": "12px",
                            "color": MUTED
                        }
                    )

                ]),

                html.Div(
                    "28 COUNTRIES",
                    style={
                        "fontSize": "11px",
                        "fontWeight": "bold",
                        "color": NAVY,
                        "backgroundColor": "#E8EEF9",
                        "padding": "8px 12px",
                        "borderRadius": "20px"
                    }
                )

            ]
        ),

        # FILTER PANEL

        html.Div(

            style={
                **CARD_STYLE,
                "display": "grid",
                "gridTemplateColumns":
                    "repeat(5, minmax(120px, 1fr))",
                "gap": "12px",
                "marginBottom": "14px"
            },

            children=[

                # YEAR

                html.Div([
                    html.Label(
                        "YEAR",
                        style={
                            "fontSize": "10px",
                            "color": MUTED
                        }
                    ),

                    dcc.Dropdown(
                        id="year-filter",
                        options=[
                            {"label": str(y), "value": y}
                            for y in YEARS
                        ],
                        value=2023,
                        clearable=False
                    )
                ]),

                # MULTIPLE AGES

                html.Div([
                    html.Label(
                        "AGE GROUPS",
                        style={
                            "fontSize": "10px",
                            "color": MUTED
                        }
                    ),

                    dcc.Dropdown(
                        id="age-filter",
                        options=[
                            {"label": a, "value": a}
                            for a in AGES
                        ],
                        value=[
                            "5-9 years",
                            "10-14 years",
                            "15-19 years"
                        ],
                        multi=True,
                        placeholder="Select ages",
                        clearable=True
                    )
                ]),

                # GENDER

                html.Div([
                    html.Label(
                        "GENDER",
                        style={
                            "fontSize": "10px",
                            "color": MUTED
                        }
                    ),

                    dcc.Dropdown(
                        id="gender-filter",
                        options=[
                            {"label": g, "value": g}
                            for g in ["Female", "Male", "Both"]
                        ],
                        value="Female",
                        clearable=False
                    )
                ]),

                # METRIC

                html.Div([
                    html.Label(
                        "METRIC",
                        style={
                            "fontSize": "10px",
                            "color": MUTED
                        }
                    ),

                    dcc.Dropdown(
                        id="metric-filter",
                        options=[
                            {
                                "label": "Prevalence (%)",
                                "value": "Percent"
                            },
                            {
                                "label": "Estimated cases",
                                "value": "Number"
                            }
                        ],
                        value="Percent",
                        clearable=False
                    )
                ]),

                # MULTIPLE COUNTRIES

                html.Div([
                    html.Label(
                        "COUNTRIES",
                        style={
                            "fontSize": "10px",
                            "color": MUTED
                        }
                    ),

                    dcc.Dropdown(
                        id="country-filter",
                        options=[
                            {"label": c, "value": c}
                            for c in COUNTRIES
                        ],
                        value=[
                            "Kyrgyzstan",
                            "Kazakhstan",
                            "Uzbekistan",
                            "Republic of Korea"
                        ],
                        multi=True,
                        placeholder="Select countries",
                        clearable=True
                    )
                ])

            ]
        ),

        # KPI STRIP

        html.Div(
            id="kpi-strip",
            style={
                "display": "grid",
                "gridTemplateColumns":
                    "repeat(3, minmax(0, 1fr))",
                "gap": "12px",
                "marginBottom": "14px"
            }
        ),

        # TOP ROW: MAP + RANKING

        html.Div(

            style={
                "display": "grid",
                "gridTemplateColumns": "1.45fr 1fr",
                "gap": "14px",
                "marginBottom": "14px"
            },

            children=[

                html.Div(
                    style=CARD_STYLE,
                    children=[

                        html.P(
                            "01  Geographic Distribution",
                            style=SECTION_TITLE
                        ),

                        dcc.Graph(
                            id="world-map",
                            config={
                                "displayModeBar": True,
                                "scrollZoom": True
                            },
                            style={"height": "390px"},
                            responsive=True
                        ),

                        html.P(
                            id="map-note",
                            style={
                                "fontSize": "10px",
                                "color": MUTED
                            }
                        )

                    ]
                ),

                html.Div(
                    style=CARD_STYLE,
                    children=[

                        html.P(
                            "02  Country Comparison",
                            style=SECTION_TITLE
                        ),

                        dcc.Graph(
                            id="country-ranking",
                            config={"displayModeBar": False},
                            style={"height": "410px"},
                            responsive=True
                        )

                    ]
                )

            ]
        ),

        # BOTTOM ROW: TREND + AGE COMPARISON

        html.Div(

            style={
                "display": "grid",
                "gridTemplateColumns": "1.2fr 1fr",
                "gap": "14px",
                "marginBottom": "14px"
            },

            children=[

                # TREND

                html.Div(
                    style=CARD_STYLE,
                    children=[

                        html.P(
                            "03  Country Trends (2019–2023)",
                            style=SECTION_TITLE
                        ),

                        dcc.Graph(
                            id="year-comparison",
                            config={"displayModeBar": False},
                            style={"height": "350px"},
                            responsive=True
                        )

                    ]
                ),

                # AGE COMPARISON

                html.Div(
                    style=CARD_STYLE,
                    children=[

                        html.P(
                            "04  Age Group Comparison",
                            style=SECTION_TITLE
                        ),

                        dcc.Graph(
                            id="age-distribution",
                            config={"displayModeBar": False},
                            style={"height": "350px"},
                            responsive=True
                        )

                    ]
                )

            ]
        ),

        # METHODOLOGY

        html.Div(
            style={
                **CARD_STYLE,
                "padding": "12px 16px"
            },

            children=[

                html.P(
                    "DATA NOTES",
                    style=SECTION_TITLE
                ),

                html.P(
                    "Source: IHME Global Burden of Disease (GBD) 2023. "
                    "Prevalence is a modelled estimate, not a count "
                    "of confirmed diagnoses. Error bars represent "
                    "95% uncertainty intervals. The Gender filter "
                    "represents the original GBD sex categories, "
                    "not gender identity. Country data contain "
                    "Male/Female estimates only; Both-sex and "
                    "All-ages country estimates are unavailable. "
                    "Only 28 selected countries are included.",

                    style={
                        "fontSize": "11px",
                        "lineHeight": "1.6",
                        "color": MUTED,
                        "margin": "0"
                    }
                )

            ]
        )

    ]
)


# =========================================================
# 8. MAP + COUNTRY RANKING + KPI
# =========================================================

@app.callback(

    Output("kpi-strip", "children"),
    Output("world-map", "figure"),
    Output("map-note", "children"),
    Output("country-ranking", "figure"),

    Input("year-filter", "value"),
    Input("age-filter", "value"),
    Input("gender-filter", "value"),
    Input("metric-filter", "value"),
    Input("country-filter", "value")

)

def update_main_visuals(
    year,
    ages,
    gender,
    metric,
    countries
):

    ages = normalize_selection(ages)
    countries = normalize_selection(countries)

    primary_age = ages[0] if ages else None

    # -----------------------------------------
    # KPI CARDS
    # -----------------------------------------

    def kpi_card(title, value, subtitle):

        return html.Div(

            style=CARD_STYLE,

            children=[

                html.P(
                    title,
                    style={
                        "fontSize": "10px",
                        "color": MUTED,
                        "margin": "0 0 5px"
                    }
                ),

                html.H3(
                    value,
                    style={
                        "fontSize": "20px",
                        "color": NAVY,
                        "margin": "0"
                    }
                ),

                html.P(
                    subtitle,
                    style={
                        "fontSize": "10px",
                        "color": MUTED,
                        "margin": "5px 0 0"
                    }
                )

            ]
        )

    kpis = [

        kpi_card(
            "SELECTED COUNTRIES",
            str(len(countries)),
            "Countries in comparison"
        ),

        kpi_card(
            "SELECTED AGE GROUPS",
            str(len(ages)),
            "Age groups in analysis"
        ),

    ]

    # -----------------------------------------
    # NO AGE OR COUNTRY SELECTION
    # -----------------------------------------

    if not ages or not countries:

        empty = empty_figure(
            "Please select at least one age group and one country."
        )

        kpis.append(
            kpi_card(
                "HIGHEST ESTIMATE",
                "N/A",
                "No selection available"
            )
        )

        return (
            kpis,
            empty,
            "Select at least one age group and one country.",
            empty
        )

    # -----------------------------------------
    # USE FIRST SELECTED AGE FOR MAP/RANKING
    # -----------------------------------------

    data = df[

        (df["location_name"].isin(countries)) &
        (df["year"] == year) &
        (df["age_name"] == primary_age) &
        (df["sex_name"] == gender) &
        (df["metric_name"] == metric)

    ].copy()

    if data.empty:

        map_fig = empty_figure(
            "No country estimates available for this selection."
        )

        ranking_fig = empty_figure(
            "No country comparison available."
        )

        highest_country = "N/A"

    else:

        highest = data.loc[
            data["display_value"].idxmax()
        ]

        highest_country = (
            highest["location_name"]
            if not data["display_value"].eq(0).all()
            else "All zero"
        )

        # -------------------------------------
        # COUNTRY RANKING
        # -------------------------------------

        ranking_data = data.sort_values(
            "display_value",
            ascending=True
        )

        ranking_fig = px.bar(

            ranking_data,

            x="display_value",
            y="location_name",

            orientation="h",

            labels={
                "display_value": (
                    "Prevalence (%)"
                    if metric == "Percent"
                    else "Estimated cases"
                ),
                "location_name": "Country"
            },

            hover_data={
                "display_value": ":.3f",
                "lower_display": ":.3f",
                "upper_display": ":.3f"
            }

        )

        ranking_fig = add_uncertainty(
            ranking_fig,
            ranking_data
        )

        style_chart(ranking_fig, 410)

        ranking_fig.update_layout(
            yaxis_title="",
            showlegend=False,
            margin=dict(l=105, r=10, t=15, b=35)
        )

        # -------------------------------------
        # WORLD MAP
        # -------------------------------------


        # -------------------------------------
        # WORLD MAP WITH NUMERICAL LABELS
        # -------------------------------------

        if data["display_value"].eq(0).all():

            map_fig = empty_figure(
                "All estimates are zero for this selection."
            )

        else:

            data["iso3"] = data["location_name"].map(ISO3)

            map_fig = px.choropleth(

                data,

                locations="iso3",
                locationmode="ISO-3",

                color="display_value",

                hover_name="location_name",

                hover_data={
                    "iso3": False,
                    "display_value": ":.3f",
                    "lower_display": ":.3f",
                    "upper_display": ":.3f"
                },

                color_continuous_scale="YlOrRd",

                labels={
                    "display_value": (
                        "Prevalence (%)"
                        if metric == "Percent"
                        else "Estimated cases"
                    ),
                    "lower_display": "Lower 95% UI",
                    "upper_display": "Upper 95% UI"
                }

            )

            # Country coordinates for numerical labels

            COUNTRY_COORDS = {
                "Australia": (-25, 134),
                "Austria": (47.5, 14),
                "Brazil": (-10, -52),
                "Canada": (56, -106),
                "China": (35, 103),
                "France": (46, 2),
                "Germany": (51, 10),
                "Ghana": (7.9, -1.2),
                "Hungary": (47.5, 19),
                "India": (22, 79),
                "Indonesia": (-2, 118),
                "Italy": (42.5, 12.5),
                "Japan": (37, 138),
                "Kazakhstan": (48, 67),
                "Kenya": (0, 37),
                "Kyrgyzstan": (41.2, 74.5),
                "Malaysia": (4, 102),
                "Nigeria": (9, 8),
                "Republic of Korea": (36, 128),
                "Russian Federation": (60, 100),
                "Rwanda": (-2, 30),
                "Spain": (40, -4),
                "Thailand": (15, 101),
                "United Kingdom": (54, -2),
                "United States of America": (39, -98),
                "Uzbekistan": (41, 64),
                "Viet Nam": (16, 107),
                "Zambia": (-13, 27)
            }

            # Prepare label data

            label_data = data.copy()

            label_data["lat"] = label_data[
                "location_name"
            ].map(lambda country: COUNTRY_COORDS[country][0])

            label_data["lon"] = label_data[
                "location_name"
            ].map(lambda country: COUNTRY_COORDS[country][1])

            # Format numerical labels

            if metric == "Percent":

                label_data["map_label"] = (
                    label_data["display_value"].map(
                        lambda value: f"{value:.2f}%"
                    )
                )

            else:

                label_data["map_label"] = (
                    label_data["display_value"].map(
                        lambda value: f"{value:,.0f}"
                    )
                )

            # Add numerical values over countries

            map_fig.add_trace(

                go.Scattergeo(

                    lat=label_data["lat"],
                    lon=label_data["lon"],

                    text=label_data["map_label"],

                    mode="text",

                    textfont=dict(
                        size=11,
                        color="#172554",
                        family="Arial"
                    ),

                    hoverinfo="text",

                    hovertext=(
                        label_data["location_name"]
                        + ": "
                        + label_data["map_label"]
                    ),

                    showlegend=False

                )

            )

            map_fig.update_geos(

                projection_type="natural earth",
                showcountries=True,
                showcoastlines=True,
                showframe=False,
                showland=True,
                landcolor="#E9EDF3",
                fitbounds="locations"

            )

            map_fig.update_layout(

                template="plotly_white",
                height=390,

                margin=dict(
                    l=0,
                    r=0,
                    t=10,
                    b=0
                ),

                coloraxis_colorbar=dict(
                    title=(
                        "Prevalence (%)"
                        if metric == "Percent"
                        else "Cases"
                    ),
                    thickness=10,
                    len=0.7
                )

            )


    kpis.append(
        kpi_card(
            "HIGHEST ESTIMATE",
            highest_country,
            f"{primary_age} · {year}"
        )
    )

    map_note = (
        f"Map and ranking use {primary_age}, "
        f"{year}, {gender}. "
        f"Selected countries: {len(countries)}. "
        "Other selected ages are shown in the age comparison below."
    )

    return kpis, map_fig, map_note, ranking_fig


# =========================================================
# 9. COUNTRY TRENDS: ALL FIVE YEARS
# =========================================================

@app.callback(

    Output("year-comparison", "figure"),

    Input("country-filter", "value"),
    Input("age-filter", "value"),
    Input("gender-filter", "value"),
    Input("metric-filter", "value")

)

def update_trend(countries, ages, gender, metric):

    countries = normalize_selection(countries)
    ages = normalize_selection(ages)

    if not countries or not ages:

        return empty_figure(
            "Select countries and age groups."
        )

    primary_age = ages[0]

    data = df[

        (df["location_name"].isin(countries)) &
        (df["age_name"] == primary_age) &
        (df["sex_name"] == gender) &
        (df["metric_name"] == metric)

    ].copy()

    if data.empty:

        return empty_figure(
            "No historical country estimates available."
        )

    fig = px.line(

        data.sort_values("year"),

        x="year",
        y="display_value",

        color="location_name",

        markers=True,

        error_y=(
            data["upper_display"] -
            data["display_value"]
        ).clip(lower=0),

        error_y_minus=(
            data["display_value"] -
            data["lower_display"]
        ).clip(lower=0),

        labels={
            "year": "Year",
            "display_value": (
                "Prevalence (%)"
                if metric == "Percent"
                else "Estimated cases"
            ),
            "location_name": "Country"
        }

    )

    style_chart(fig, 350)

    fig.update_layout(
        xaxis=dict(
            tickmode="linear",
            dtick=1
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=-0.25
        )
    )

    return fig


# =========================================================
# 10. MULTIPLE AGE GROUP COMPARISON
# =========================================================

@app.callback(

    Output("age-distribution", "figure"),

    Input("year-filter", "value"),
    Input("age-filter", "value"),
    Input("country-filter", "value"),
    Input("gender-filter", "value"),
    Input("metric-filter", "value")

)

def update_age_comparison(
    year,
    ages,
    countries,
    gender,
    metric
):

    ages = normalize_selection(ages)
    countries = normalize_selection(countries)

    if not ages or not countries:

        return empty_figure(
            "Select age groups and countries."
        )

    data = df[

        (df["location_name"].isin(countries)) &
        (df["year"] == year) &
        (df["age_name"].isin(ages)) &
        (df["sex_name"] == gender) &
        (df["metric_name"] == metric)

    ].copy()

    if data.empty:

        return empty_figure(
            "No age-specific country estimates available. "
            "Check whether All ages or Both is selected."
        )

    data["age_name"] = pd.Categorical(
        data["age_name"],
        categories=AGE_ORDER,
        ordered=True
    )

    data = data.sort_values("age_name")

    fig = px.bar(

        data,

        x="age_name",
        y="display_value",

        color="location_name",

        barmode="group",

        error_y=(
            data["upper_display"] -
            data["display_value"]
        ).clip(lower=0),

        error_y_minus=(
            data["display_value"] -
            data["lower_display"]
        ).clip(lower=0),

        labels={
            "age_name": "Age group",
            "display_value": (
                "Prevalence (%)"
                if metric == "Percent"
                else "Estimated cases"
            ),
            "location_name": "Country"
        }

    )

    style_chart(fig, 350)

    fig.update_layout(
        xaxis_tickangle=-35,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=-0.35
        )
    )

    return fig


# =========================================================
# 11. START DASHBOARD
# =========================================================

if __name__ == "__main__":

    print("\nADHD RESEARCH DASHBOARD")
    print("Records:", len(df))
    print("Countries:", len(COUNTRIES))
    print("Years:", YEARS)
    print("Open: http://127.0.0.1:8050")

    app.run(debug=True)
