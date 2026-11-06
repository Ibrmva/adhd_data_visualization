import base64
import io
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, State, no_update
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


# =========================================================
# 1. LOAD ORIGINAL CSV FILES
# =========================================================

DATA_FOLDER = Path(__file__).resolve().parent / "data"

FILES = [
    "2019-23.csv",
    "all ages_19-23.csv",
    "geo_distribution.csv",
]

REQUIRED_COLUMNS = [
    "measure_name", "cause_name", "location_id", "location_name",
    "sex_id", "sex_name", "age_id", "age_name", "metric_id",
    "metric_name", "year", "val", "upper", "lower",
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

base_df = pd.concat(frames, ignore_index=True)


# =========================================================
# 2. DATA PROCESSING
# =========================================================

def classify_condition(cause_name):
    if pd.isna(cause_name):
        return None

    cause = str(cause_name).lower().strip()

    if (
        "attention-deficit" in cause
        or "attention deficit" in cause
        or "adhd" in cause
    ):
        return "ADHD"

    if "autism" in cause or "asd" in cause:
        return "ASD"

    return None


def prepare_dataframe(data):
    data = data.copy()
    data.columns = data.columns.str.strip()

    for col in REQUIRED_COLUMNS:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    for col in ["year", "val", "lower", "upper"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["year"] = data["year"].astype("Int64")
    data["condition"] = data["cause_name"].apply(classify_condition)

    data = data[
        (data["measure_name"].astype(str).str.strip() == "Prevalence")
        & (data["condition"] == "ADHD")
    ].copy()

    # Keep original GBD values and prepare display values.
    data["display_value"] = data["val"]
    data["lower_display"] = data["lower"]
    data["upper_display"] = data["upper"]

    percent_mask = data["metric_name"].astype(str).str.strip() == "Percent"

    data.loc[percent_mask, "display_value"] *= 100
    data.loc[percent_mask, "lower_display"] *= 100
    data.loc[percent_mask, "upper_display"] *= 100

    return data


base_df = prepare_dataframe(base_df)

if base_df.empty:
    raise ValueError("No ADHD Prevalence records found in original CSV files.")


# Check duplicates in original data.
KEYS = [
    "location_id", "sex_id", "age_id", "year",
    "metric_id", "measure_id", "cause_id"
]

available_keys = [key for key in KEYS if key in base_df.columns]

if available_keys and base_df.duplicated(subset=available_keys).any():
    raise ValueError("Duplicate observations detected in the original data.")


def get_dashboard_data(uploaded_records):
    """
    If uploaded CSV data exists, use it as the active dataset.
    Otherwise, use the original five CSV files.
    """

    if uploaded_records:
        uploaded_df = pd.DataFrame(uploaded_records)

        if not uploaded_df.empty:
            return prepare_dataframe(uploaded_df)

    return base_df.copy()


# =========================================================
# 3. COUNTRY AND AGE DATA
# =========================================================

ISO3 = {
    "Australia": "AUS", "Austria": "AUT", "Brazil": "BRA",
    "Canada": "CAN", "China": "CHN", "France": "FRA",
    "Germany": "DEU", "Ghana": "GHA", "Hungary": "HUN",
    "India": "IND", "Indonesia": "IDN", "Italy": "ITA",
    "Japan": "JPN", "Kazakhstan": "KAZ", "Kenya": "KEN",
    "Kyrgyzstan": "KGZ", "Malaysia": "MYS", "Nigeria": "NGA",
    "Republic of Korea": "KOR", "Russian Federation": "RUS",
    "Rwanda": "RWA", "Spain": "ESP", "Thailand": "THA",
    "United Kingdom": "GBR", "United States of America": "USA",
    "Uzbekistan": "UZB", "Viet Nam": "VNM", "Zambia": "ZMB",
}

COUNTRY_COORDS = {
    "Australia": (-25, 134), "Austria": (47.5, 14),
    "Brazil": (-10, -52), "Canada": (56, -106),
    "China": (35, 103), "France": (46, 2),
    "Germany": (51, 10), "Ghana": (7.9, -1.2),
    "Hungary": (47.5, 19), "India": (22, 79),
    "Indonesia": (-2, 118), "Italy": (42.5, 12.5),
    "Japan": (37, 138), "Kazakhstan": (48, 67),
    "Kenya": (0, 37), "Kyrgyzstan": (41.2, 74.5),
    "Malaysia": (4, 102), "Nigeria": (9, 8),
    "Republic of Korea": (36, 128), "Russian Federation": (60, 100),
    "Rwanda": (-2, 30), "Spain": (40, -4),
    "Thailand": (15, 101), "United Kingdom": (54, -2),
    "United States of America": (39, -98), "Uzbekistan": (41, 64),
    "Viet Nam": (16, 107), "Zambia": (-13, 27),
}

AGE_ORDER = [
    "<28 days", "1-5 months", "6-11 months", "<1 year",
    "1-4 years", "2-4 years", "5-9 years", "10-14 years",
    "15-19 years", "20-24 years", "25-29 years", "30-34 years",
    "35-39 years", "40-44 years", "45-49 years", "50-54 years",
    "55-59 years", "60-64 years", "65-69 years", "70-74 years",
    "75-79 years", "80+ years", "All ages",
]


def get_age_options(data):
    available = data["age_name"].dropna().astype(str).unique().tolist()

    ordered = [age for age in AGE_ORDER if age in available]
    remaining = sorted([age for age in available if age not in ordered])

    return ordered + remaining


def get_available_countries(data):
    return sorted(
        data.loc[
            data["location_name"] != "Global",
            "location_name"
        ].dropna().astype(str).unique().tolist()
    )


def get_available_years(data):
    return sorted(data["year"].dropna().astype(int).unique().tolist())


def get_available_genders(data):
    return sorted(data["sex_name"].dropna().astype(str).unique().tolist())


def get_available_metrics(data):
    return sorted(data["metric_name"].dropna().astype(str).unique().tolist())


def get_available_diseases(data):
    order = ["ADHD"]
    available = data["condition"].dropna().unique().tolist()
    return [disease for disease in order if disease in available]


YEARS = get_available_years(base_df)
AGES = get_age_options(base_df)
COUNTRIES = get_available_countries(base_df)

if not YEARS:
    raise ValueError("No valid years found in the original data.")


# =========================================================
# 4. LANGUAGE
# =========================================================

T = {
    "EN": {
        "title": "ADHD Research Dashboard",
        "subtitle": "Global Burden of Disease 2023",
        "upload": "UPLOAD DATA",
        "upload_btn": "Upload ADHD CSV",
        "no_upload": "No files uploaded yet.",
        "disease": "DISEASE",
        "from": "FROM YEAR",
        "to": "TO YEAR",
        "ages": "AGE GROUPS",
        "gender": "GENDER (GBD sex categories)",
        "metric": "METRIC",
        "countries": "COUNTRIES",
        "compare": "Compare both",
        "geo": "01  Geographic Distribution",
        "ranking": "02  Country Comparison",
        "trends": "03  Prevalence Trends",
        "age_chart": "04  Age Group Comparison",
        "notes": "DATA NOTES",
        "report": "RESEARCH REPORT",
        "summary": "Executive Summary",
        "stats": "Selected Statistics",
        "findings": "Key Findings",
        "method": "Methodology and Limitations",
        "download": "Download PDF Report",
        "selected_countries": "Selected countries",
        "selected_ages": "Selected age groups",
        "highest": "Highest estimate",
        "no_data": "No data for selected filters.",
        "source": "Source: IHME Global Burden of Disease (GBD) 2023.",
        "period": "Selected period",
        "gender_selected": "Selected gender categories (GBD sex categories)",
        "metric_selected": "Selected metric",
        "records": "Observations",
        "mean": "Mean estimate",
        "min": "Minimum estimate",
        "max": "Maximum estimate",
        "high_country": "Highest country estimate",
        "low_country": "Lowest country estimate",
        "no_country": "No country comparison is available.",
        "trend_up": "The estimate increased over the selected period.",
        "trend_down": "The estimate decreased over the selected period.",
        "trend_same": "No change between the first and last available years.",
        "trend_na": "Not enough observations to calculate a trend.",
        "intro": "This report summarizes the data currently selected in the dashboard.",
        "model_warning": "Prevalence estimates are modelled population estimates, not counts of confirmed diagnoses.",
        "sex_note": "The interface label is Gender, but GBD provides biological sex categories, not gender identity.",
        "ui_note": "Intervals show the 95% uncertainty interval around the ADHD estimate.",
        "map_compare": "Comparison mode: the map is disabled because two conditions cannot share one country colour scale.",
        "map_age": "Map age group",
        "country": "Country",
        "disease_label": "Disease",
        "year": "Year",
        "age_group": "Age group",
        "prevalence": "Prevalence (%)",
        "cases": "Estimated cases",
        "upload_help": "Supported format: IHME GBD CSV containing ADHD prevalence records and the required columns.",
        "try_filters": "Try changing the filters or uploading the relevant CSV.",
        "select_country_age": "Select countries and age groups.",
        "no_history": "No historical estimates available.",
        "no_age": "No age-specific estimates available.",
        "map_none": "No mapped country estimates available.",
        "zero": "All estimates are zero for this selection.",
        "map_excluded": "Countries without a configured ISO-3 code are excluded from the map.",
        "filter_countries": "Countries in comparison",
        "filter_ages": "Age groups in analysis",
        "highest_none": "No matching records",
        "upload_success": "Successfully loaded",
        "upload_error": "Errors",
        "all": "All",
    },

    "RU": {
        "title": "Исследовательская панель: СДВГ (ADHD)",
        "subtitle": "Глобальное бремя заболеваний, 2023",
        "upload": "ЗАГРУЗКА ДАННЫХ",
        "upload_btn": "Загрузить CSV: СДВГ (ADHD)",
        "no_upload": "Файлы ещё не загружены.",
        "disease": "ЗАБОЛЕВАНИЕ",
        "from": "С ГОДА",
        "to": "ПО ГОД",
        "ages": "ВОЗРАСТНЫЕ ГРУППЫ",
        "gender": "ГЕНДЕР (категории пола GBD)",
        "metric": "ПОКАЗАТЕЛЬ",
        "countries": "СТРАНЫ",
        "compare": "Сравнить оба",
        "geo": "01  Географическое распределение",
        "ranking": "02  Сравнение стран",
        "trends": "03  Динамика распространённости",
        "age_chart": "04  Сравнение возрастных групп",
        "notes": "ПРИМЕЧАНИЯ К ДАННЫМ",
        "report": "ИССЛЕДОВАТЕЛЬСКИЙ ОТЧЁТ",
        "summary": "Краткое содержание",
        "stats": "Выбранные статистические показатели",
        "findings": "Основные результаты",
        "method": "Методология и ограничения",
        "download": "Скачать PDF-отчёт",
        "selected_countries": "Выбранные страны",
        "selected_ages": "Выбранные возрастные группы",
        "highest": "Наибольшая оценка",
        "no_data": "Нет данных для выбранных фильтров.",
        "source": "Источник: IHME Global Burden of Disease (GBD) 2023.",
        "period": "Выбранный период",
        "gender_selected": "Выбранные категории (GBD: пол)",
        "metric_selected": "Выбранный показатель",
        "records": "Количество наблюдений",
        "mean": "Средняя оценка",
        "min": "Минимальная оценка",
        "max": "Максимальная оценка",
        "high_country": "Наибольшая оценка среди стран",
        "low_country": "Наименьшая оценка среди стран",
        "no_country": "Сравнение стран недоступно.",
        "trend_up": "Оценка увеличилась за выбранный период.",
        "trend_down": "Оценка снизилась за выбранный период.",
        "trend_same": "Между первым и последним доступными годами изменений нет.",
        "trend_na": "Недостаточно наблюдений для расчёта динамики.",
        "intro": "В отчёте представлены данные, выбранные в панели фильтров.",
        "model_warning": "Показатели распространённости являются модельными оценками для населения, а не количеством подтверждённых диагнозов.",
        "sex_note": "Фильтр подписан как Gender, однако GBD предоставляет категории биологического пола, а не гендерную идентичность.",
        "ui_note": "Интервалы отражают 95% интервал неопределённости оценки распространённости ADHD.",
        "map_compare": "Режим сравнения: карта отключена, поскольку две нозологии нельзя отобразить одной цветовой шкалой.",
        "map_age": "Возрастная группа карты",
        "country": "Страна",
        "disease_label": "Заболевание",
        "year": "Год",
        "age_group": "Возрастная группа",
        "prevalence": "Распространённость (%)",
        "cases": "Расчётное количество случаев",
        "upload_help": "Поддерживается CSV IHME GBD с данными о распространённости ADHD и обязательными столбцами.",
        "try_filters": "Измените фильтры или загрузите соответствующий CSV.",
        "select_country_age": "Выберите страны и возрастные группы.",
        "no_history": "Исторические оценки отсутствуют.",
        "no_age": "Нет оценок для выбранных возрастных групп.",
        "map_none": "Нет географических данных для карты.",
        "zero": "Все оценки для этого выбора равны нулю.",
        "map_excluded": "Страны без настроенного кода ISO-3 исключены с карты.",
        "filter_countries": "Стран в сравнении",
        "filter_ages": "Возрастных групп в анализе",
        "highest_none": "Нет подходящих записей",
        "upload_success": "Успешно загружено",
        "upload_error": "Ошибки",
        "all": "Все",
    },
}


def tr(lang, key):
    return T.get(lang, T["EN"]).get(key, T["EN"].get(key, key))


# =========================================================
# 5. HELPERS
# =========================================================

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
    "boxShadow": "0 2px 6px rgba(15,23,42,0.035)",
}

SECTION_TITLE = {
    "fontSize": "15px",
    "fontWeight": "700",
    "color": NAVY,
    "margin": "0 0 8px 0",
}

LABEL_STYLE = {
    "fontSize": "10px",
    "color": MUTED,
    "fontWeight": "bold",
    "display": "block",
    "marginBottom": "5px",
}


def empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=.5,
        y=.5,
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

    return f"{value:.3f}%" if metric == "Percent" else f"{value:,.0f}"


def style_chart(fig, height=350):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=35, r=15, t=35, b=40),
        font=dict(family="Arial", size=11, color="#334155"),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    return fig


def normalize_selection(values):
    if values is None:
        return []

    return [values] if isinstance(values, str) else values


def selection_label(values):
    values = normalize_selection(values)
    return ", ".join(map(str, values)) if values else "All"


def get_year_bounds(start_year, end_year):
    if start_year is None or end_year is None:
        return None, None

    return min(start_year, end_year), max(start_year, end_year)


def apply_common_filters(data, start_year, end_year, ages, countries, gender, metric):
    data = data.copy()

    start_year, end_year = get_year_bounds(start_year, end_year)

    if start_year is not None:
        data = data[data["year"].between(start_year, end_year)]

    ages = normalize_selection(ages)
    countries = normalize_selection(countries)
    genders = normalize_selection(gender)

    if ages:
        data = data[data["age_name"].isin(ages)]

    if countries:
        data = data[data["location_name"].isin(countries)]

    if genders:
        data = data[data["sex_name"].isin(genders)]

    if metric:
        data = data[data["metric_name"] == metric]

    return data.copy()


def add_uncertainty(fig, data):
    fig.update_traces(
        error_y=dict(
            type="data",
            symmetric=False,
            array=(
                data["upper_display"] - data["display_value"]
            ).clip(lower=0).tolist(),
            arrayminus=(
                data["display_value"] - data["lower_display"]
            ).clip(lower=0).tolist()
        )
    )
    return fig


def make_kpi_card(title, value, subtitle):
    return html.Div(
        style=CARD_STYLE,
        children=[
            html.P(
                title,
                style={"fontSize": "10px", "color": MUTED, "margin": "0 0 5px"}
            ),
            html.H3(
                value,
                style={"fontSize": "20px", "color": NAVY, "margin": "0"}
            ),
            html.P(
                subtitle,
                style={"fontSize": "10px", "color": MUTED, "margin": "5px 0 0"}
            ),
        ]
    )


def filtered_report_data(
    uploaded_records, disease, start_year, end_year,
    ages, gender, metric, countries
):
    data = get_dashboard_data(uploaded_records)

    filtered = apply_common_filters(
        data,
        start_year,
        end_year,
        normalize_selection(ages),
        normalize_selection(countries),
        gender,
        metric
    )

    if disease == "Compare":
        filtered = filtered[
            filtered["condition"].isin(["ADHD", "ASD"])
        ]
    else:
        filtered = filtered[filtered["condition"] == disease]

    return filtered.copy()


# =========================================================
# 6. REPORT COMPONENTS
# =========================================================


def get_research_sections(lang):
    """Return concise, evidence-based findings for the ADHD research report."""
    is_ru = lang == "RU"

    def source_rows(filename, **filters):
        frame = base_df.copy()
        if "source_file" in frame.columns:
            frame = frame[frame["source_file"].astype(str) == filename]
        for column, value in filters.items():
            if isinstance(value, (list, tuple, set)):
                frame = frame[frame[column].isin(value)]
            else:
                frame = frame[frame[column] == value]
        return frame

    def one_row(frame, **filters):
        for column, value in filters.items():
            frame = frame[frame[column] == value]
        return None if frame.empty else frame.iloc[0]

    def pct(value):
        return "N/A" if value is None or pd.isna(value) else f"{float(value):.2f}%"

    def million(value):
        return "N/A" if value is None or pd.isna(value) else f"{float(value) / 1_000_000:.2f} млн" if is_ru else f"{float(value) / 1_000_000:.2f} million"

    # Global prevalence and sex estimates, 2023.
    global_all = source_rows(
        "all ages_19-23.csv", location_name="Global", age_name="All ages",
        year=2023, metric_name="Percent"
    )
    both = one_row(global_all, sex_name="Both")
    male = one_row(global_all, sex_name="Male")
    female = one_row(global_all, sex_name="Female")
    global_number = one_row(source_rows(
        "all ages_19-23.csv", location_name="Global", age_name="All ages",
        year=2023, metric_name="Number"
    ), sex_name="Both")

    global_value = pct(both["display_value"] if both is not None else None)
    global_cases = million(global_number["display_value"] if global_number is not None else None)
    male_value = pct(male["display_value"] if male is not None else None)
    female_value = pct(female["display_value"] if female is not None else None)

    # Age distribution, global Both, 2023.
    age_2023 = source_rows(
        "2019-23.csv", location_name="Global", sex_name="Both",
        year=2023, metric_name="Percent"
    ).copy()
    age_2023 = age_2023.sort_values("display_value", ascending=False)
    top_age = age_2023.iloc[0] if not age_2023.empty else None
    next_ages = age_2023.head(3) if not age_2023.empty else age_2023
    age_text = "; ".join(
        f"{row['age_name']}: {pct(row['display_value'])}"
        for _, row in next_ages.iterrows()
    ) if not age_2023.empty else ("Нет данных" if is_ru else "No data")

    # Country distribution, 28-country subset, 2023, ages 10-14.
    country = source_rows(
        "geo_distribution.csv", year=2023, age_name="10-14 years",
        metric_name="Percent"
    ).copy()
    country_findings = []
    for sex, label in [("Male", "Мальчики/мужчины" if is_ru else "Male"),
                       ("Female", "Девочки/женщины" if is_ru else "Female")]:
        subset = country[country["sex_name"] == sex].sort_values("display_value")
        if not subset.empty:
            low = subset.iloc[0]
            high = subset.iloc[-1]
            country_findings.append(
                f"{label}: {high['location_name']} — {pct(high['display_value'])}; "
                f"{low['location_name']} — {pct(low['display_value'])}"
            )
    geography_text = "\n".join(country_findings) if country_findings else ("Нет данных" if is_ru else "No data")

    # Global time trend, both sexes, all ages, 2019-2023.
    trend_pct = source_rows(
        "all ages_19-23.csv", location_name="Global", sex_name="Both",
        age_name="All ages", metric_name="Percent"
    ).sort_values("year")
    trend_num = source_rows(
        "all ages_19-23.csv", location_name="Global", sex_name="Both",
        age_name="All ages", metric_name="Number"
    ).sort_values("year")
    if len(trend_pct) >= 2:
        trend_pct_text = f"{int(trend_pct.iloc[0]['year'])}: {pct(trend_pct.iloc[0]['display_value'])} → {int(trend_pct.iloc[-1]['year'])}: {pct(trend_pct.iloc[-1]['display_value'])}"
    else:
        trend_pct_text = "Нет данных" if is_ru else "No data"
    if len(trend_num) >= 2:
        trend_num_text = f"{int(trend_num.iloc[0]['year'])}: {million(trend_num.iloc[0]['display_value'])} → {int(trend_num.iloc[-1]['year'])}: {million(trend_num.iloc[-1]['display_value'])}"
    else:
        trend_num_text = "Нет данных" if is_ru else "No data"

    if is_ru:
        sections = [
            {
                "title": "1. Определение ADHD",
                "finding": "ADHD — расстройство нейроразвития, связанное с устойчивыми трудностями внимания, гиперактивностью и/или импульсивностью.",
                "note": "Симптомы должны влиять на повседневное функционирование; отдельный симптом не является диагнозом.",
                "links": [("Определение и клиническая информация", "https://www.rcpsych.ac.uk/mental-health/translations/russian/adhd-in-adults")],
            },
            {
                "title": "2. Основные симптомы",
                "finding": "Три группы: невнимательность; гиперактивность; импульсивность.",
                "note": "Для диагностики важны устойчивость симптомов, начало в детстве и трудности в нескольких сферах жизни.",
                "links": [("Симптомы и диагностика (RCPsych)", "https://www.rcpsych.ac.uk/mental-health/translations/russian/adhd-in-adults")],
            },
            {
                "title": "3. Глобальная распространённость",
                "finding": f"GBD 2023: {global_value} среди всего населения мира; расчётное число случаев — {global_cases}.",
                "note": "Модельная распространённость, а не доля людей, получивших диагноз в 2023 году.",
                "links": [("IHME GBD Results Tool", "https://ghdx.healthdata.org/gbd-results-tool")],
            },
            {
                "title": "4. Возрастное распределение и возраст первого диагноза",
                "finding": f"GBD 2023, глобально: {age_text}. В исследовании Уэльса средний возраст первого зарегистрированного диагноза: мужчины — 10,9 года, женщины — 12,6 года.",
                "note": "Возрастная распространённость и возраст первого диагноза — разные показатели. Данные Уэльса относятся к когорте 3–30 лет, зарегистрированной в 2000–2019 годах.",
                "links": [("Исследование Уэльса, 2024", "https://pubmed.ncbi.nlm.nih.gov/38864317/"), ("Исходные возрастные данные GBD", "https://ghdx.healthdata.org/gbd-results-tool")],
            },
            {
                "title": "5. Различия по полу",
                "finding": f"GBD 2023, все возрасты: Male — {male_value}; Female — {female_value}.",
                "note": "GBD использует категории биологического пола. Это модельные оценки распространённости, а не показатели гендерной идентичности или вероятности постановки диагноза.",
                "links": [("IHME GBD Results Tool", "https://ghdx.healthdata.org/gbd-results-tool"), ("RCPsych: распознавание ADHD у женщин", "https://www.rcpsych.ac.uk/mental-health/translations/russian/adhd-in-adults")],
            },
            {
                "title": "6. Три клинических проявления ADHD",
                "finding": "Метаанализ детей младше 12 лет: невнимательное — 33,2% (95% ДИ 27,6–39,3); гиперактивно-импульсивное — 30,3% (23,8–37,7); комбинированное — 31,4% (24,6–39,1).",
                "note": "Оценки объединялись отдельно; их нельзя складывать в 100%.",
                "links": [("Метаанализ, 2023", "https://pmc.ncbi.nlm.nih.gov/articles/PMC10120242/")],
            },
            {
                "title": "7. Сопутствующие состояния: ADHD и ASD",
                "finding": "Систематический обзор 2024 года включил 9 исследований (548 участников с ADHD); клинически значимые ASD-симптомы встречались в диапазоне 15–64,3%.",
                "note": "Разброс связан с различиями выборок и методов. Это диапазон симптомов, а не объединённая оценка подтверждённого двойного диагноза. В GBD CSV совместная диагностика не измеряется.",
                "links": [("Систематический обзор ADHD–ASD, 2024", "https://link.springer.com/article/10.1007/s40489-024-00443-4"), ("RCPsych: сопутствующие состояния", "https://www.rcpsych.ac.uk/mental-health/translations/russian/adhd-in-adults")],
            },
            {
                "title": "8. Географическое распределение",
                "finding": "2023, возраст 10–14 лет, 28 стран:\n" + geography_text,
                "note": "Это только страны, присутствующие в geo_distribution.csv. Сравниваются модельные оценки, отдельно по категориям Male и Female.",
                "links": [("Полная база GBD", "https://ghdx.healthdata.org/gbd-results-tool")],
            },
            {
                "title": "9. Динамика распространённости и диагностики",
                "finding": f"GBD global, все возрасты: распространённость {trend_pct_text}; расчётное число случаев {trend_num_text}. США: доля детей 5–17 лет, когда-либо получавших диагноз ADHD, выросла с 9,9% (2019) до 13,8% (2025).",
                "note": "GBD показывает модельную распространённость, а CDC — когда-либо установленный диагноз. Это разные показатели. CSV не содержат числа официальных диагнозов или количества недиагностированных людей. В Англии зарегистрированная распространённость на июнь 2025 года составила 1,19%; разрыв с эпидемиологическими референсными оценками составлял от 28,71% (мужчины 18–24 лет) до 93,51% (люди 65+). Это разрыв диагностического охвата, а не прямое число недиагностированных людей.",
                "links": [("CDC, США, 2019–2025", "https://www.cdc.gov/mmwr/volumes/75/wr/mm7542a2.htm"), ("Англия: зарегистрированные диагнозы, 2000–2025 (публикация 2026)", "https://pmc.ncbi.nlm.nih.gov/articles/PMC13453955/"), ("IHME GBD Results Tool", "https://ghdx.healthdata.org/gbd-results-tool")],
            },
        ]
    else:
        sections = [
            {"title": "1. ADHD definition", "finding": "ADHD is a neurodevelopmental disorder associated with persistent difficulties with attention, hyperactivity and/or impulsivity.", "note": "Symptoms must affect everyday functioning; an isolated symptom is not a diagnosis.", "links": [("Definition and clinical information", "https://www.rcpsych.ac.uk/mental-health/translations/russian/adhd-in-adults")]},
            {"title": "2. Main symptoms", "finding": "Three symptom groups: inattention, hyperactivity and impulsivity.", "note": "Diagnosis considers persistence, childhood onset and impairment across multiple areas of life.", "links": [("Symptoms and diagnosis (RCPsych)", "https://www.rcpsych.ac.uk/mental-health/translations/russian/adhd-in-adults")]},
            {"title": "3. Global prevalence", "finding": f"GBD 2023: {global_value} of the global population; estimated cases: {global_cases}.", "note": "Modelled prevalence, not the share of people diagnosed during 2023.", "links": [("IHME GBD Results Tool", "https://ghdx.healthdata.org/gbd-results-tool")]},
            {"title": "4. Age distribution and age at diagnosis", "finding": f"Global GBD 2023: {age_text}. In a Wales study, mean age at first recorded diagnosis was 10.9 years for males and 12.6 years for females.", "note": "Age-specific prevalence and age at first diagnosis are different measures. Wales estimates refer to a cohort aged 3–30 with records from 2000–2019.", "links": [("Wales study, 2024", "https://pubmed.ncbi.nlm.nih.gov/38864317/"), ("GBD age data", "https://ghdx.healthdata.org/gbd-results-tool")]},
            {"title": "5. Sex differences", "finding": f"GBD 2023, all ages: Male — {male_value}; Female — {female_value}.", "note": "GBD reports biological sex categories. These are modelled prevalence estimates, not gender identity or diagnosis probability.", "links": [("IHME GBD Results Tool", "https://ghdx.healthdata.org/gbd-results-tool"), ("RCPsych: recognition in women", "https://www.rcpsych.ac.uk/mental-health/translations/russian/adhd-in-adults")]},
            {"title": "6. Three ADHD presentations", "finding": "Meta-analysis of children under 12: inattentive 33.2% (95% CI 27.6–39.3); hyperactive-impulsive 30.3% (23.8–37.7); combined 31.4% (24.6–39.1).", "note": "Estimates were pooled separately and should not be added to 100%.", "links": [("Meta-analysis, 2023", "https://pmc.ncbi.nlm.nih.gov/articles/PMC10120242/")]},
            {"title": "7. Co-occurring conditions: ADHD and ASD", "finding": "A 2024 systematic review included 9 studies (548 participants with ADHD); clinically significant ASD symptoms ranged from 15% to 64.3%.", "note": "The range reflects differences in samples and methods. It is a range of symptom findings, not a pooled rate of confirmed dual diagnosis. GBD CSVs do not measure co-diagnosis.", "links": [("ADHD–ASD systematic review, 2024", "https://link.springer.com/article/10.1007/s40489-024-00443-4"), ("RCPsych: co-occurring conditions", "https://www.rcpsych.ac.uk/mental-health/translations/russian/adhd-in-adults")]},
            {"title": "8. Geographic distribution", "finding": "2023, ages 10–14, 28 countries:\n" + geography_text, "note": "Only countries included in geo_distribution.csv are represented. Modelled estimates are shown separately for Male and Female.", "links": [("Full GBD database", "https://ghdx.healthdata.org/gbd-results-tool")]},
            {"title": "9. Prevalence and diagnosis trends", "finding": f"Global GBD, all ages: prevalence {trend_pct_text}; estimated cases {trend_num_text}. United States: the share of children aged 5–17 ever diagnosed with ADHD increased from 9.9% (2019) to 13.8% (2025).", "note": "GBD reports modelled prevalence; CDC reports ever-diagnosed prevalence. These are different measures. The CSVs do not contain official diagnosis counts or the number of undiagnosed people. In England, recorded prevalence in June 2025 was 1.19%; the gap from epidemiological reference estimates ranged from 28.71% (males aged 18–24) to 93.51% (people aged 65+). This is a diagnostic-coverage gap, not a direct count of undiagnosed people.", "links": [("CDC, US, 2019–2025", "https://www.cdc.gov/mmwr/volumes/75/wr/mm7542a2.htm"), ("England: recorded diagnoses, 2000–2025 (published 2026)", "https://pmc.ncbi.nlm.nih.gov/articles/PMC13453955/"), ("IHME GBD Results Tool", "https://ghdx.healthdata.org/gbd-results-tool")]},
        ]
    return sections


def make_research_area_sections(lang):
    """Compact bilingual research sections; details are available through source links."""
    is_ru = lang == "RU"
    cards = []
    for section in get_research_sections(lang):
        links = [
            html.A(
                label + " ↗", href=url, target="_blank", rel="noopener noreferrer",
                style={
                    "display": "inline-block", "margin": "5px 7px 0 0",
                    "padding": "6px 10px", "border": "1px solid #CBD5E1",
                    "borderRadius": "7px", "color": "#1D4ED8",
                    "textDecoration": "none", "fontSize": "11px", "fontWeight": "600",
                }
            ) for label, url in section["links"]
        ]
        cards.append(html.Details([
            html.Summary(section["title"], style={
                "cursor": "pointer", "fontWeight": "700", "color": NAVY,
                "padding": "12px 14px", "listStylePosition": "inside"
            }),
            html.Div([
                html.P(section["finding"], style={
                    "fontWeight": "600", "whiteSpace": "pre-line",
                    "lineHeight": "1.55", "margin": "5px 0 8px"
                }),
                html.P(section["note"], style={
                    "fontSize": "11px", "color": MUTED,
                    "lineHeight": "1.5", "margin": "0 0 7px"
                }),
                html.Div(links),
            ], style={"padding": "4px 15px 14px", "borderTop": f"1px solid {BORDER}"})
        ], open=False, style={
            "border": f"1px solid {BORDER}", "borderRadius": "9px",
            "marginBottom": "8px", "backgroundColor": "white"
        }))

    return html.Div([
        html.H4("Исследовательские направления ADHD" if is_ru else "ADHD Research Areas",
                style={"color": NAVY, "margin": "18px 0 10px"}),
        html.P("Краткие результаты; дополнительные сведения открываются по ссылкам в каждом разделе." if is_ru else "Concise findings; additional information is available through each section's source links.",
               style={"fontSize": "12px", "color": MUTED, "marginBottom": "12px"}),
        *cards
    ])


def make_report_components(data, lang, disease, start_year, end_year, ages, gender, metric, countries):
    t = lambda key: tr(lang, key)

    if data.empty:
        summary = html.Div(
            t("no_data"),
            style={"color": MUTED, "padding": "10px 0"}
        )
    else:
        data = data.dropna(subset=["display_value"]).copy()

        if data.empty:
            summary = html.Div(
                t("no_data"),
                style={"color": MUTED, "padding": "10px 0"}
            )
        else:
            values = data["display_value"]
            avg = values.mean()
            low = values.min()
            high = values.max()
            max_row = data.loc[values.idxmax()]

            country_data = data[
                data["location_name"] != "Global"
            ].groupby(
                ["location_name", "condition"], as_index=False
            )["display_value"].mean()

            if not country_data.empty:
                high_country = country_data.loc[country_data["display_value"].idxmax()]
                low_country = country_data.loc[country_data["display_value"].idxmin()]
                country_lines = [
                    f"{t('high_country')}: {high_country['location_name']} "
                    f"({high_country['condition']}) — "
                    f"{format_value(high_country['display_value'], metric)}",
                    f"{t('low_country')}: {low_country['location_name']} "
                    f"({low_country['condition']}) — "
                    f"{format_value(low_country['display_value'], metric)}",
                ]
            else:
                country_lines = [t("no_country")]

            years = data["year"].dropna().astype(int)
            period = f"{years.min()}–{years.max()}" if len(years) else "N/A"
            selected_countries = ", ".join(countries) if countries else t("all")
            selected_ages = ", ".join(ages) if ages else t("all")

            yearly = (
                data.groupby("year")["display_value"]
                .mean().dropna().sort_index()
            )

            if len(yearly) >= 2:
                first, last = yearly.iloc[0], yearly.iloc[-1]
                if last > first:
                    trend_text = t("trend_up")
                elif last < first:
                    trend_text = t("trend_down")
                else:
                    trend_text = t("trend_same")
            else:
                trend_text = t("trend_na")

            gender_label = (
                ", ".join(normalize_selection(gender))
                if normalize_selection(gender)
                else t("all")
            )

            summary = html.Div([
                html.H4(t("summary"), style={"color": NAVY, "marginBottom": "6px"}),
                html.P(t("intro"), style={"color": MUTED, "fontSize": "12px"}),
                html.P(
                    f"{t('period')}: {period} | "
                    f"{t('gender_selected')}: {gender_label} | "
                    f"{t('metric_selected')}: {metric}",
                    style={"fontSize": "12px"}
                ),
                html.P(
                    f"{t('selected_countries')}: {selected_countries}",
                    style={"fontSize": "12px"}
                ),
                html.P(
                    f"{t('selected_ages')}: {selected_ages}",
                    style={"fontSize": "12px"}
                ),
                html.H4(t("stats"), style={"color": NAVY, "marginTop": "16px"}),
                html.Ul([
                    html.Li(f"{t('records')}: {len(data):,}"),
                    html.Li(f"{t('mean')}: {format_value(avg, metric)}"),
                    html.Li(f"{t('min')}: {format_value(low, metric)}"),
                    html.Li(f"{t('max')}: {format_value(high, metric)}"),
                    html.Li(
                        f"{t('highest')}: {max_row['location_name']} "
                        f"({max_row['condition']}) — "
                        f"{format_value(max_row['display_value'], metric)}"
                    ),
                ]),
                html.H4(t("findings"), style={"color": NAVY, "marginTop": "16px"}),
                html.Ul([html.Li(x) for x in country_lines + [trend_text]]),
                html.H4(t("method"), style={"color": NAVY, "marginTop": "16px"}),
                html.P(t("source"), style={"fontSize": "11px"}),
                html.P(t("model_warning"), style={"fontSize": "11px"}),
                html.P(t("sex_note"), style={"fontSize": "11px"}),
                html.P(t("ui_note"), style={"fontSize": "11px"}),
            ])

    return html.Div([
        summary,
        html.Hr(style={"border": "0", "borderTop": f"1px solid {BORDER}", "margin": "18px 0"}),
        make_research_area_sections(lang)
    ])


# =========================================================
# 7. PDF REPORT
# =========================================================

def find_pdf_font():
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/local/share/fonts/DejaVuSans.ttf",
    ]

    for font_path in candidates:
        if Path(font_path).exists():
            try:
                pdfmetrics.registerFont(
                    TTFont("DashboardUnicode", font_path)
                )
                return "DashboardUnicode"
            except Exception:
                pass

    return "Helvetica"


def create_pdf_report(
    data, lang, disease, start_year, end_year,
    ages, gender, metric, countries
):
    t = lambda key: tr(lang, key)

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=17 * mm,
        bottomMargin=17 * mm
    )

    font_name = find_pdf_font()
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DashTitle",
        parent=styles["Title"],
        fontName=font_name,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#172554"),
        fontSize=17,
        leading=22
    )

    heading_style = ParagraphStyle(
        "DashHeading",
        parent=styles["Heading2"],
        fontName=font_name,
        textColor=colors.HexColor("#172554"),
        spaceBefore=10,
        spaceAfter=5
    )

    body_style = ParagraphStyle(
        "DashBody",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=9,
        leading=13,
        spaceAfter=5
    )

    story = [
        Paragraph(t("title"), title_style),
        Spacer(1, 5 * mm),
        Paragraph(t("intro"), body_style),

        Paragraph(
            f"<b>{t('disease_label')}:</b> {disease}",
            body_style
        ),

        Paragraph(
            f"<b>{t('period')}:</b> {start_year}–{end_year}",
            body_style
        ),

        Paragraph(
            f"<b>{t('gender_selected')}:</b> {selection_label(gender)}",
            body_style
        ),

        Paragraph(
            f"<b>{t('metric_selected')}:</b> {metric}",
            body_style
        ),

        Paragraph(
            f"<b>{t('selected_ages')}:</b> "
            f"{', '.join(ages) if ages else 'All'}",
            body_style
        ),

        Paragraph(
            f"<b>{t('selected_countries')}:</b> "
            f"{', '.join(countries) if countries else 'All'}",
            body_style
        ),
    ]

    if data.empty:
        story += [
            Spacer(1, 5 * mm),
            Paragraph(t("no_data"), body_style)
        ]

    else:
        vals = data["display_value"].dropna()

        max_row = data.loc[data["display_value"].idxmax()]
        min_row = data.loc[data["display_value"].idxmin()]

        yearly = (
            data.groupby("year")["display_value"]
            .mean()
            .dropna()
            .sort_index()
        )

        if len(yearly) >= 2:
            trend = (
                t("trend_up") if yearly.iloc[-1] > yearly.iloc[0]
                else t("trend_down") if yearly.iloc[-1] < yearly.iloc[0]
                else t("trend_same")
            )
        else:
            trend = t("trend_na")

        rows = [
            [t("stats"), "Value"],
            [t("records"), f"{len(data):,}"],
            [t("mean"), format_value(vals.mean(), metric)],
            [t("min"), format_value(vals.min(), metric)],
            [t("max"), format_value(vals.max(), metric)],
            [
                t("highest"),
                f"{max_row['location_name']} ({max_row['condition']}): "
                f"{format_value(max_row['display_value'], metric)}"
            ],
            [
                t("min"),
                f"{min_row['location_name']} ({min_row['condition']}): "
                f"{format_value(min_row['display_value'], metric)}"
            ],
        ]

        table = Table(
            rows,
            colWidths=[65 * mm, 105 * mm],
            repeatRows=1
        )

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#172554")),
            ("FONTNAME", (0, 0), (-1, -1), font_name),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#D8DEE9")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))

        story += [
            Spacer(1, 5 * mm),
            table,
            Paragraph(t("findings"), heading_style),
            Paragraph(trend, body_style),
            Paragraph(t("method"), heading_style),
            Paragraph(t("source"), body_style),
            Paragraph(t("model_warning"), body_style),
            Paragraph(t("sex_note"), body_style),
            Paragraph(t("ui_note"), body_style),
        ]

    # =====================================================
    # COUNTRY COMPARISON APPENDIX (28-country GBD file)
    # This appendix is independent of the interactive filters.
    # =====================================================
    country_source = base_df.copy()
    if "source_file" in country_source.columns:
        country_source = country_source[
            country_source["source_file"].astype(str) == "geo_distribution.csv"
        ].copy()

    country_source = country_source[
        (country_source["year"] == 2023)
        & (country_source["age_name"].astype(str) == "10-14 years")
        & (country_source["metric_name"].astype(str) == "Percent")
        & (country_source["sex_name"].astype(str).isin(["Male", "Female"]))
    ].copy()

    if not country_source.empty:
        is_ru = lang == "RU"
        appendix_title = (
            "Географическое сравнение: 28 стран" if is_ru
            else "Geographic comparison: 28 countries"
        )
        appendix_note = (
            "Сравнение модельной распространённости СДВГ в 2023 году среди детей 10–14 лет. "
            "Значения представлены отдельно для Male и Female; в скобках указан 95% интервал неопределённости. "
            "Выборка включает только 28 стран, представленных в исходном geo_distribution.csv. "
            "Категории Male/Female в GBD отражают половые категории источника и не являются показателем гендерной идентичности."
            if is_ru else
            "Modelled ADHD prevalence in 2023 among children aged 10–14. "
            "Male and Female estimates are shown separately; 95% uncertainty intervals are in parentheses. "
            "The comparison includes only the 28 countries available in the source geo_distribution.csv. "
            "GBD Male/Female categories represent the source's sex categories and do not measure gender identity."
        )

        story += [
            Spacer(1, 7 * mm),
            Paragraph(appendix_title, heading_style),
            Paragraph(appendix_note, body_style),
        ]

        country_rows = []
        for country in sorted(country_source["location_name"].dropna().unique()):
            row_data = country_source[country_source["location_name"] == country]
            sex_values = {}
            for sex in ["Male", "Female"]:
                selected = row_data[row_data["sex_name"] == sex]
                if selected.empty:
                    sex_values[sex] = ("N/A", "N/A")
                else:
                    item = selected.iloc[0]
                    estimate = f"{item['display_value']:.2f}%"
                    interval = (
                        f"({item['lower_display']:.2f}–"
                        f"{item['upper_display']:.2f})"
                    )
                    sex_values[sex] = (estimate, interval)

            country_rows.append([
                str(country),
                sex_values["Male"][0], sex_values["Male"][1],
                sex_values["Female"][0], sex_values["Female"][1],
            ])

        country_header = (
            ["Страна", "Male (%)", "95% интервал", "Female (%)", "95% интервал"]
            if is_ru else
            ["Country", "Male (%)", "95% UI", "Female (%)", "95% UI"]
        )
        country_table = Table(
            [country_header] + country_rows,
            colWidths=[42 * mm, 25 * mm, 37 * mm, 25 * mm, 37 * mm],
            repeatRows=1
        )
        country_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#172554")),
            ("FONTNAME", (0, 0), (-1, -1), font_name),
            ("FONTSIZE", (0, 0), (-1, -1), 6.5),
            ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#D8DEE9")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story += [country_table, Spacer(1, 5 * mm)]

        trend_countries = [
            "Kyrgyzstan", "Kazakhstan", "Uzbekistan", "Republic of Korea"
        ]
        trend_source = base_df.copy()
        if "source_file" in trend_source.columns:
            trend_source = trend_source[
                trend_source["source_file"].astype(str) == "geo_distribution.csv"
            ].copy()
        trend_source = trend_source[
            trend_source["location_name"].isin(trend_countries)
            & (trend_source["age_name"].astype(str) == "10-14 years")
            & (trend_source["metric_name"].astype(str) == "Percent")
            & (trend_source["sex_name"].astype(str).isin(["Male", "Female"]))
            & (trend_source["year"].between(2019, 2023))
        ].copy()

        if not trend_source.empty:
            trend_title = (
                "Динамика в отдельных странах, 2019–2023" if is_ru
                else "Selected-country trends, 2019–2023"
            )
            story.append(Paragraph(trend_title, heading_style))
            trend_header = (
                ["Страна", "Категория", "2019", "2020", "2021", "2022", "2023"]
                if is_ru else
                ["Country", "Category", "2019", "2020", "2021", "2022", "2023"]
            )
            trend_rows = []
            for country in trend_countries:
                for sex in ["Male", "Female"]:
                    subset = trend_source[
                        (trend_source["location_name"] == country)
                        & (trend_source["sex_name"] == sex)
                    ]
                    if subset.empty:
                        continue
                    by_year = {
                        int(item["year"]): item["display_value"]
                        for _, item in subset.iterrows()
                    }
                    country_label = (
                        "Республика Корея" if is_ru and country == "Republic of Korea"
                        else country
                    )
                    trend_rows.append([
                        country_label, sex,
                        *[f"{by_year[y]:.2f}%" if y in by_year else "—"
                          for y in range(2019, 2024)]
                    ])

            trend_table = Table(
                [trend_header] + trend_rows,
                colWidths=[48 * mm, 23 * mm, 20 * mm, 20 * mm, 20 * mm, 20 * mm, 20 * mm],
                repeatRows=1
            )
            trend_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF9")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#172554")),
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#D8DEE9")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(trend_table)

        source_note = (
            "Источник: IHME Global Burden of Disease (GBD), предоставленный файл geo_distribution.csv. "
            "Оценки распространённости являются модельными; они не равны числу подтверждённых диагнозов. "
            "Проценты и интервалы неопределённости рассчитаны из значений Percent исходного CSV."
            if is_ru else
            "Source: IHME Global Burden of Disease (GBD), supplied geo_distribution.csv. "
            "Prevalence estimates are modelled and are not counts of confirmed diagnoses. "
            "Percentages and uncertainty intervals are displayed from the source CSV Percent values."
        )
        story += [Spacer(1, 3 * mm), Paragraph(source_note, body_style)]

    # Concise research-area findings are included in the PDF export too.
    research_title = "Исследовательские направления" if lang == "RU" else "Research areas"
    story += [Spacer(1, 7 * mm), Paragraph(research_title, heading_style)]
    research_rows = [["Направление" if lang == "RU" else "Research area",
                      "Ключевой результат" if lang == "RU" else "Key finding"]]
    for section in get_research_sections(lang):
        links_text = " | ".join(
            f'<link href="{url}" color="#2563EB">{label}</link>'
            for label, url in section["links"]
        )
        cell_text = (
            f"{section['finding'].replace(chr(10), '<br/>')}<br/><font color='#64748B' size='7'>{section['note']}</font>"
            f"<br/>{links_text}"
        )
        research_rows.append([
            Paragraph(section["title"], body_style),
            Paragraph(cell_text, body_style),
        ])
    research_table = Table(
        research_rows, colWidths=[48 * mm, 122 * mm], repeatRows=1
    )
    research_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#172554")),
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#D8DEE9")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(research_table)

    doc.build(story)

    return buffer.getvalue()


# =========================================================
# 8. DASH APP + LAYOUT
# =========================================================

app = Dash(__name__)
app.title = "ADHD Research Dashboard"


def section_card(title_id, title, graph_id, height):
    return html.Div(
        style=CARD_STYLE,
        children=[
            html.P(title, id=title_id, style=SECTION_TITLE),
            dcc.Graph(
                id=graph_id,
                config={"displayModeBar": False},
                style={"height": f"{height}px"},
                responsive=True
            )
        ]
    )


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
                "gap": "12px",
                "marginBottom": "14px"
            },

            children=[
                html.Div([
                    html.H2(
                        tr("EN", "title"),
                        id="main-title",
                        style={
                            "margin": "0",
                            "color": NAVY,
                            "fontSize": "24px"
                        }
                    ),

                    html.P(
                        tr("EN", "subtitle"),
                        id="main-subtitle",
                        style={
                            "margin": "5px 0 0",
                            "fontSize": "12px",
                            "color": MUTED
                        }
                    ),
                ]),

                dcc.Dropdown(
                    id="language-filter",
                    options=[
                        {"label": "English", "value": "EN"},
                        {"label": "Русский", "value": "RU"}
                    ],
                    value="EN",
                    clearable=False,
                    searchable=False,
                    style={"width": "140px"}
                ),
            ]
        ),

        # UPLOAD
        html.Div(
            style={**CARD_STYLE, "marginBottom": "14px"},
            children=[
                html.P(
                    "UPLOAD DATA",
                    id="upload-title",
                    style=SECTION_TITLE
                ),

                dcc.Upload(
                    id="upload-data",
                    multiple=True,
                    accept=".csv",

                    children=html.Div(
                        "＋ Upload ADHD CSV",
                        id="upload-button-label",
                        style={
                            "backgroundColor": NAVY,
                            "color": "white",
                            "padding": "13px 22px",
                            "borderRadius": "8px",
                            "fontSize": "13px",
                            "fontWeight": "bold",
                            "cursor": "pointer",
                            "display": "inline-block"
                        }
                    ),

                    style={
                        "display": "inline-block",
                        "cursor": "pointer"
                    }
                ),

                html.Div(
                    id="upload-status",
                    children=tr("EN", "no_upload"),
                    style={
                        "fontSize": "11px",
                        "color": MUTED,
                        "marginTop": "8px"
                    }
                ),

                html.P(
                    tr("EN", "upload_help"),
                    id="upload-help",
                    style={
                        "fontSize": "10px",
                        "color": MUTED,
                        "margin": "10px 0 0"
                    }
                ),

                # MEMORY STORAGE: avoids browser sessionStorage quota issue.
                dcc.Store(
                    id="uploaded-data",
                    storage_type="memory",
                    data=[]
                ),
            ]
        ),

        # FILTERS
        html.Div(
            style={
                **CARD_STYLE,
                "display": "grid",
                "gridTemplateColumns": "repeat(6, minmax(115px, 1fr))",
                "gap": "12px",
                "marginBottom": "14px"
            },

            children=[
                html.Div([
                    html.Label(
                        "DISEASE",
                        id="disease-label",
                        style=LABEL_STYLE
                    ),

                    dcc.Dropdown(
                        id="disease-filter",
                        options=[
                            {"label": "ADHD", "value": "ADHD"}
                        ],
                        value="ADHD",
                        clearable=False
                    )
                ]),

                html.Div([
                    html.Label(
                        "FROM YEAR",
                        id="from-label",
                        style=LABEL_STYLE
                    ),

                    dcc.Dropdown(
                        id="year-start-filter",
                        options=[
                            {"label": str(y), "value": y}
                            for y in YEARS
                        ],
                        value=min(YEARS),
                        clearable=False
                    )
                ]),

                html.Div([
                    html.Label(
                        "TO YEAR",
                        id="to-label",
                        style=LABEL_STYLE
                    ),

                    dcc.Dropdown(
                        id="year-end-filter",
                        options=[
                            {"label": str(y), "value": y}
                            for y in YEARS
                        ],
                        value=max(YEARS),
                        clearable=False
                    )
                ]),

                html.Div([
                    html.Label(
                        "AGE GROUPS",
                        id="ages-label",
                        style=LABEL_STYLE
                    ),

                    dcc.Dropdown(
                        id="age-filter",
                        options=[
                            {"label": a, "value": a}
                            for a in AGES
                        ],
                        value=[
                            a for a in [
                                "5-9 years",
                                "10-14 years",
                                "15-19 years"
                            ] if a in AGES
                        ],
                        multi=True,
                        placeholder="Select ages",
                        clearable=True
                    )
                ]),

                html.Div([
                    html.Label(
                        "GENDER (GBD sex categories)",
                        id="gender-label",
                        style=LABEL_STYLE
                    ),

                    dcc.Dropdown(
                        id="gender-filter",
                        options=[
                            {"label": g, "value": g}
                            for g in get_available_genders(base_df)
                        ],
                        value=get_available_genders(base_df),
                        multi=True,
                        placeholder="Select GBD categories",
                        clearable=True
                    )
                ]),

                html.Div([
                    html.Label(
                        "METRIC",
                        id="metric-label",
                        style=LABEL_STYLE
                    ),

                    dcc.Dropdown(
                        id="metric-filter",
                        options=[
                            {"label": "Prevalence (%)", "value": "Percent"},
                            {"label": "Estimated cases", "value": "Number"}
                        ],
                        value=(
                            "Percent"
                            if "Percent" in get_available_metrics(base_df)
                            else get_available_metrics(base_df)[0]
                        ),
                        clearable=False
                    )
                ]),
            ]
        ),

        # COUNTRIES
        html.Div(
            style={**CARD_STYLE, "marginBottom": "14px"},
            children=[
                html.Label(
                    "COUNTRIES",
                    id="countries-label",
                    style=LABEL_STYLE
                ),

                dcc.Dropdown(
                    id="country-filter",
                    options=[
                        {"label": c, "value": c}
                        for c in COUNTRIES
                    ],
                    value=COUNTRIES,
                    multi=True,
                    placeholder="Select countries",
                    clearable=True
                ),
            ]
        ),

        # KPI
        html.Div(
            id="kpi-strip",
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(3, minmax(0, 1fr))",
                "gap": "12px",
                "marginBottom": "14px"
            }
        ),

        # MAP + RANKING
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
                            id="map-title",
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
                        ),
                    ]
                ),

                section_card(
                    "ranking-title",
                    "02  Country Comparison",
                    "country-ranking",
                    410
                ),
            ]
        ),

        # TRENDS + AGE
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1.2fr 1fr",
                "gap": "14px",
                "marginBottom": "14px"
            },

            children=[
                section_card(
                    "trends-title",
                    "03  Prevalence Trends",
                    "year-comparison",
                    350
                ),

                section_card(
                    "age-chart-title",
                    "04  Age Group Comparison",
                    "age-distribution",
                    350
                ),
            ]
        ),

        # NOTES
        html.Div(
            style={**CARD_STYLE, "marginBottom": "14px"},
            children=[
                html.P(
                    "DATA NOTES",
                    id="notes-title",
                    style=SECTION_TITLE
                ),

                html.P(
                    id="methodology-note",
                    children=(
                        "Source: IHME Global Burden of Disease (GBD) 2023. "
                        "Prevalence is a modelled estimate, not a count of confirmed diagnoses. "
                        "Error bars represent 95% uncertainty intervals. The GBD filter is labelled gender in the interface, but represents the original GBD sex categories "
                        "rather than gender identity. This dashboard analyses ADHD only."
                    ),
                    style={
                        "fontSize": "11px",
                        "lineHeight": "1.6",
                        "color": MUTED,
                        "margin": "0"
                    }
                ),
            ]
        ),

        # REPORT
        html.Div(
            style={**CARD_STYLE, "marginBottom": "20px"},
            children=[
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "gap": "10px",
                        "flexWrap": "wrap"
                    },

                    children=[
                        html.P(
                            "RESEARCH REPORT",
                            id="report-title",
                            style=SECTION_TITLE
                        ),

                        html.Button(
                            "Download PDF Report",
                            id="download-report-button",
                            n_clicks=0,
                            style={
                                "backgroundColor": NAVY,
                                "color": "white",
                                "border": "0",
                                "borderRadius": "7px",
                                "padding": "10px 16px",
                                "fontWeight": "bold",
                                "cursor": "pointer"
                            }
                        ),
                    ]
                ),

                html.Div(id="research-report"),
                dcc.Download(id="download-report-pdf"),
            ]
        ),
    ]
)


# =========================================================
# 9. LANGUAGE SWITCHER
# =========================================================

@app.callback(
    Output("main-title", "children"),
    Output("main-subtitle", "children"),
    Output("upload-title", "children"),
    Output("upload-button-label", "children"),
    Output("upload-help", "children"),
    Output("disease-label", "children"),
    Output("from-label", "children"),
    Output("to-label", "children"),
    Output("ages-label", "children"),
    Output("gender-label", "children"),
    Output("metric-label", "children"),
    Output("countries-label", "children"),
    Output("map-title", "children"),
    Output("ranking-title", "children"),
    Output("trends-title", "children"),
    Output("age-chart-title", "children"),
    Output("notes-title", "children"),
    Output("report-title", "children"),
    Output("download-report-button", "children"),
    Output("methodology-note", "children"),

    Input("language-filter", "value"),
)
def update_language(lang):
    keys = [
        "title", "subtitle", "upload", "upload_btn", "upload_help",
        "disease", "from", "to", "ages", "gender", "metric", "countries",
        "geo", "ranking", "trends", "age_chart", "notes", "report",
        "download",
    ]

    values = [tr(lang, key) for key in keys]

    note = " ".join([
        tr(lang, "source"),
        tr(lang, "model_warning"),
        tr(lang, "sex_note"),
        tr(lang, "ui_note")
    ])

    return (*values, note)


# =========================================================
# 10. CSV UPLOAD CALLBACK
# =========================================================

@app.callback(
    Output("uploaded-data", "data"),
    Output("upload-status", "children"),

    Input("upload-data", "contents"),

    State("upload-data", "filename"),
    State("language-filter", "value"),

    prevent_initial_call=True,
)
def upload_csv(contents, filenames, lang):

    if not contents:
        return [], tr(lang, "no_upload")

    if not isinstance(contents, list):
        contents = [contents]

    if not isinstance(filenames, list):
        filenames = [filenames]

    uploaded_frames = []
    successful = []
    errors = []

    for content, filename in zip(contents, filenames):
        try:
            _, content_string = content.split(",", 1)

            decoded = base64.b64decode(content_string)

            uploaded_file = pd.read_csv(
                io.StringIO(decoded.decode("utf-8-sig"))
            )

            uploaded_file.columns = uploaded_file.columns.str.strip()

            missing = [
                col for col in REQUIRED_COLUMNS
                if col not in uploaded_file.columns
            ]

            if missing:
                raise ValueError(
                    "Missing columns: " + ", ".join(missing)
                )

            prepared = prepare_dataframe(uploaded_file)

            if prepared.empty:
                raise ValueError(
                    "No ADHD Prevalence records found."
                )

            prepared["source_file"] = filename

            uploaded_frames.append(prepared)

            successful.append(
                f"{filename}: {len(prepared):,} records"
            )

        except Exception as error:
            errors.append(f"{filename}: {error}")

    # A new upload batch REPLACES the previous uploaded dataset.
    # Multiple files in the same upload are combined.
    if uploaded_frames:
        combined = pd.concat(
            uploaded_frames,
            ignore_index=True
        ).drop_duplicates()

        records = json.loads(
            combined.to_json(
                orient="records",
                date_format="iso"
            )
        )

    else:
        records = []

    messages = []

    if successful:
        messages.append(
            tr(lang, "upload_success") + ": " + " | ".join(successful)
        )

    if errors:
        messages.append(
            tr(lang, "upload_error") + ": " + " | ".join(errors)
        )

    if not messages:
        return [], "No valid files were uploaded."

    return records, " ".join(messages)


# =========================================================
# 11. DYNAMIC FILTER OPTIONS AFTER UPLOAD
# =========================================================

@app.callback(
    Output("disease-filter", "options"),
    Output("disease-filter", "value"),

    Output("country-filter", "options"),
    Output("country-filter", "value"),

    Output("age-filter", "options"),
    Output("age-filter", "value"),

    Output("year-start-filter", "options"),
    Output("year-start-filter", "value"),

    Output("year-end-filter", "options"),
    Output("year-end-filter", "value"),

    Output("gender-filter", "options"),
    Output("gender-filter", "value"),

    Output("metric-filter", "options"),
    Output("metric-filter", "value"),

    Input("uploaded-data", "data"),
)
def update_filter_options(uploaded_records):

    data = get_dashboard_data(uploaded_records)

    diseases = get_available_diseases(data)
    countries = get_available_countries(data)
    ages = get_age_options(data)
    years = get_available_years(data)
    genders = get_available_genders(data)
    metrics = get_available_metrics(data)

    disease_options = [
        {"label": disease, "value": disease}
        for disease in diseases
    ]

    if len(diseases) > 1:
        disease_options.append({
            "label": "Compare both",
            "value": "Compare"
        })

    default_disease = (
        "ADHD" if "ADHD" in diseases
        else diseases[0] if diseases
        else None
    )

    country_options = [
        {"label": country, "value": country}
        for country in countries
    ]

    age_options = [
        {"label": age, "value": age}
        for age in ages
    ]

    year_options = [
        {"label": str(year), "value": year}
        for year in years
    ]

    gender_options = [
        {"label": gender, "value": gender}
        for gender in genders
    ]

    metric_options = [
        {
            "label": (
                "Prevalence (%)"
                if metric == "Percent"
                else "Estimated cases"
                if metric == "Number"
                else metric
            ),
            "value": metric
        }
        for metric in metrics
    ]

    default_metric = (
        "Percent" if "Percent" in metrics
        else metrics[0] if metrics
        else None
    )

    start_year = min(years) if years else None
    end_year = max(years) if years else None

    return (
        disease_options,
        default_disease,

        country_options,
        countries,

        age_options,
        ages,

        year_options,
        start_year,

        year_options,
        end_year,

        gender_options,
        genders,

        metric_options,
        default_metric,
    )


# =========================================================
# 12. KPI + MAP + COUNTRY RANKING
# =========================================================

@app.callback(
    Output("kpi-strip", "children"),
    Output("world-map", "figure"),
    Output("map-note", "children"),
    Output("country-ranking", "figure"),

    Input("uploaded-data", "data"),
    Input("disease-filter", "value"),
    Input("year-start-filter", "value"),
    Input("year-end-filter", "value"),
    Input("age-filter", "value"),
    Input("gender-filter", "value"),
    Input("metric-filter", "value"),
    Input("country-filter", "value"),
    Input("language-filter", "value"),
)
def update_main_visuals(
    uploaded_records, disease, start_year, end_year,
    ages, gender, metric, countries, lang
):
    ages = normalize_selection(ages)
    countries = normalize_selection(countries)

    conditions = (
        ["ADHD", "ASD"]
        if disease == "Compare"
        else [disease]
    )

    filtered = apply_common_filters(
        get_dashboard_data(uploaded_records),
        start_year,
        end_year,
        ages,
        countries,
        gender,
        metric
    )

    filtered = filtered[
        filtered["condition"].isin(conditions)
    ].copy()

    start_year, end_year = get_year_bounds(
        start_year,
        end_year
    )

    kpis = [
        make_kpi_card(
            tr(lang, "selected_countries"),
            str(len(countries)),
            tr(lang, "filter_countries")
        ),

        make_kpi_card(
            tr(lang, "selected_ages"),
            str(len(ages)),
            tr(lang, "filter_ages")
        ),
    ]

    if filtered.empty:
        kpis.append(
            make_kpi_card(
                tr(lang, "highest"),
                "N/A",
                tr(lang, "highest_none")
            )
        )

        empty = empty_figure(tr(lang, "no_data"))

        return (
            kpis,
            empty,
            tr(lang, "try_filters"),
            empty
        )

    filtered = filtered.dropna(subset=["display_value"])

    if filtered.empty:
        highest = None
    else:
        highest = filtered.loc[
            filtered["display_value"].idxmax()
        ]

    highest_label = (
        f"{highest['location_name']} ({highest['condition']})"
        if disease == "Compare"
        else highest["location_name"]
    )

    kpis.append(
        make_kpi_card(
            tr(lang, "highest"),
            highest_label,
            format_value(highest["display_value"], metric)
        )
    )

    # COUNTRY RANKING
    # Aggregate duplicate observations by country and condition.
    ranking_data = (
        filtered.groupby(
            ["location_name", "condition"],
            as_index=False
        )
        .agg(
            display_value=("display_value", "mean"),
            lower_display=("lower_display", "mean"),
            upper_display=("upper_display", "mean")
        )
    )

    if disease == "Compare":
        ranking_data["series_label"] = (
            ranking_data["location_name"]
            + " · "
            + ranking_data["condition"]
        )

        ranking_data = ranking_data.sort_values(
            "display_value",
            ascending=True
        )

        ranking_fig = px.bar(
            ranking_data,
            x="display_value",
            y="series_label",
            orientation="h",
            color="condition",
            barmode="group",

            error_x=(
                ranking_data["upper_display"]
                - ranking_data["display_value"]
            ).clip(lower=0),

            error_x_minus=(
                ranking_data["display_value"]
                - ranking_data["lower_display"]
            ).clip(lower=0),

            labels={
                "display_value": (
                    tr(lang, "prevalence")
                    if metric == "Percent"
                    else tr(lang, "cases")
                ),
                "series_label": (
                    f"{tr(lang, 'country')} · "
                    f"{tr(lang, 'disease_label')}"
                ),
                "condition": tr(lang, "disease_label")
            }
        )

        style_chart(ranking_fig, 410)

        ranking_fig.update_layout(
            yaxis_title="",
            margin=dict(l=130, r=10, t=15, b=35)
        )

    else:
        ranking_data = ranking_data.sort_values(
            "display_value",
            ascending=True
        )

        ranking_fig = px.bar(
            ranking_data,
            x="display_value",
            y="location_name",
            orientation="h",

            error_x=(
                ranking_data["upper_display"]
                - ranking_data["display_value"]
            ).clip(lower=0),

            error_x_minus=(
                ranking_data["display_value"]
                - ranking_data["lower_display"]
            ).clip(lower=0),

            labels={
                "display_value": (
                    tr(lang, "prevalence")
                    if metric == "Percent"
                    else tr(lang, "cases")
                ),
                "location_name": tr(lang, "country")
            }
        )

        style_chart(ranking_fig, 410)

        ranking_fig.update_layout(
            yaxis_title="",
            showlegend=False,
            margin=dict(l=105, r=10, t=15, b=35)
        )

    # MAP
    if disease == "Compare":
        map_fig = empty_figure(tr(lang, "map_compare"))
        map_note = tr(lang, "map_compare")

    else:
        map_data = filtered.copy()

        if ages:
            map_data = map_data[
                map_data["age_name"] == ages[0]
            ].copy()

        # Aggregate selected sex categories and other dimensions
        # to obtain one value per country for the map.
        map_data = (
            map_data.groupby(
                "location_name",
                as_index=False
            )
            .agg(
                display_value=("display_value", "mean"),
                lower_display=("lower_display", "mean"),
                upper_display=("upper_display", "mean")
            )
        )

        map_data["iso3"] = map_data["location_name"].map(ISO3)
        map_data = map_data.dropna(subset=["iso3"])

        if map_data.empty:
            map_fig = empty_figure(tr(lang, "map_none"))

        elif map_data["display_value"].eq(0).all():
            map_fig = empty_figure(tr(lang, "zero"))

        else:
            map_fig = px.choropleth(
                map_data,
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
                        tr(lang, "prevalence")
                        if metric == "Percent"
                        else tr(lang, "cases")
                    ),
                    "lower_display": "Lower 95% UI",
                    "upper_display": "Upper 95% UI"
                }
            )

            label_data = map_data[
                map_data["location_name"].isin(COUNTRY_COORDS)
            ].copy()

            label_data["lat"] = label_data[
                "location_name"
            ].map(lambda c: COUNTRY_COORDS[c][0])

            label_data["lon"] = label_data[
                "location_name"
            ].map(lambda c: COUNTRY_COORDS[c][1])

            label_data["map_label"] = label_data[
                "display_value"
            ].map(
                lambda v: (
                    f"{v:.2f}%"
                    if metric == "Percent"
                    else f"{v:,.0f}"
                )
            )

            map_fig.add_trace(
                go.Scattergeo(
                    lat=label_data["lat"],
                    lon=label_data["lon"],
                    text=label_data["map_label"],
                    mode="text",
                    textfont=dict(
                        size=11,
                        color=NAVY,
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
                margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_colorbar=dict(
                    title=(
                        tr(lang, "prevalence")
                        if metric == "Percent"
                        else tr(lang, "cases")
                    ),
                    thickness=10,
                    len=.7
                )
            )

        gender_text = selection_label(gender)

        map_note = (
            f"{tr(lang, 'map_age')}: "
            f"{ages[0] if ages else 'All'}; "
            f"{start_year}–{end_year}; "
            f"{gender_text}. "
            f"{tr(lang, 'map_excluded')}"
        )

    return kpis, map_fig, map_note, ranking_fig


# =========================================================
# 13. COUNTRY TRENDS
# =========================================================

@app.callback(
    Output("year-comparison", "figure"),

    Input("uploaded-data", "data"),
    Input("disease-filter", "value"),
    Input("year-start-filter", "value"),
    Input("year-end-filter", "value"),
    Input("country-filter", "value"),
    Input("age-filter", "value"),
    Input("gender-filter", "value"),
    Input("metric-filter", "value"),
    Input("language-filter", "value"),
)
def update_trend(
    uploaded_records, disease, start_year, end_year,
    countries, ages, gender, metric, lang
):
    countries = normalize_selection(countries)
    ages = normalize_selection(ages)

    if not countries or not ages:
        return empty_figure(
            tr(lang, "select_country_age")
        )

    filtered = apply_common_filters(
        get_dashboard_data(uploaded_records),
        start_year,
        end_year,
        ages[:1],
        countries,
        gender,
        metric
    )

    if disease == "Compare":
        filtered = filtered[
            filtered["condition"].isin(["ADHD", "ASD"])
        ]
    else:
        filtered = filtered[
            filtered["condition"] == disease
        ]

    if filtered.empty:
        return empty_figure(tr(lang, "no_history"))

    # Aggregate selected sex categories to avoid duplicate lines.
    filtered = (
        filtered.groupby(
            ["year", "location_name", "condition"],
            as_index=False
        )
        .agg(
            display_value=("display_value", "mean"),
            lower_display=("lower_display", "mean"),
            upper_display=("upper_display", "mean")
        )
    )

    filtered = filtered.sort_values("year")

    labels = {
        "year": tr(lang, "year"),
        "display_value": (
            tr(lang, "prevalence")
            if metric == "Percent"
            else tr(lang, "cases")
        ),
        "condition": tr(lang, "disease_label"),
        "location_name": tr(lang, "country")
    }

    if disease == "Compare":
        filtered["series_label"] = (
            filtered["location_name"]
            + " · "
            + filtered["condition"]
        )

        fig = px.line(
            filtered,
            x="year",
            y="display_value",
            color="condition",
            line_dash="location_name",
            line_group="series_label",
            markers=True,

            error_y=(
                filtered["upper_display"]
                - filtered["display_value"]
            ).clip(lower=0),

            error_y_minus=(
                filtered["display_value"]
                - filtered["lower_display"]
            ).clip(lower=0),

            labels=labels
        )

    else:
        fig = px.line(
            filtered,
            x="year",
            y="display_value",
            color="location_name",
            markers=True,

            error_y=(
                filtered["upper_display"]
                - filtered["display_value"]
            ).clip(lower=0),

            error_y_minus=(
                filtered["display_value"]
                - filtered["lower_display"]
            ).clip(lower=0),

            labels=labels
        )

    style_chart(fig, 350)

    fig.update_layout(
        xaxis=dict(tickmode="linear", dtick=1),
        hovermode="x unified",
        legend=dict(orientation="h", y=-.25)
    )

    return fig


# =========================================================
# 14. AGE GROUP COMPARISON
# =========================================================

@app.callback(
    Output("age-distribution", "figure"),

    Input("uploaded-data", "data"),
    Input("disease-filter", "value"),
    Input("year-start-filter", "value"),
    Input("year-end-filter", "value"),
    Input("age-filter", "value"),
    Input("country-filter", "value"),
    Input("gender-filter", "value"),
    Input("metric-filter", "value"),
    Input("language-filter", "value"),
)
def update_age_comparison(
    uploaded_records, disease, start_year, end_year,
    ages, countries, gender, metric, lang
):
    ages = normalize_selection(ages)
    countries = normalize_selection(countries)

    if not ages or not countries:
        return empty_figure(
            tr(lang, "select_country_age")
        )

    filtered = apply_common_filters(
        get_dashboard_data(uploaded_records),
        start_year,
        end_year,
        ages,
        countries,
        gender,
        metric
    )

    if disease == "Compare":
        filtered = filtered[
            filtered["condition"].isin(["ADHD", "ASD"])
        ]
    else:
        filtered = filtered[
            filtered["condition"] == disease
        ]

    if filtered.empty:
        return empty_figure(tr(lang, "no_age"))

    # Aggregate across selected sex categories.
    filtered = (
        filtered.groupby(
            ["age_name", "location_name", "condition"],
            as_index=False
        )
        .agg(
            display_value=("display_value", "mean"),
            lower_display=("lower_display", "mean"),
            upper_display=("upper_display", "mean")
        )
    )

    age_categories = get_age_options(
        get_dashboard_data(uploaded_records)
    )

    filtered["age_name"] = pd.Categorical(
        filtered["age_name"],
        categories=age_categories,
        ordered=True
    )

    filtered = filtered.sort_values("age_name")

    labels = {
        "age_name": tr(lang, "age_group"),
        "display_value": (
            tr(lang, "prevalence")
            if metric == "Percent"
            else tr(lang, "cases")
        ),
        "condition": tr(lang, "disease_label"),
        "location_name": tr(lang, "country")
    }

    if disease == "Compare":
        fig = px.bar(
            filtered,
            x="age_name",
            y="display_value",
            color="condition",
            pattern_shape="location_name",
            barmode="group",

            error_y=(
                filtered["upper_display"]
                - filtered["display_value"]
            ).clip(lower=0),

            error_y_minus=(
                filtered["display_value"]
                - filtered["lower_display"]
            ).clip(lower=0),

            labels=labels
        )

    else:
        fig = px.bar(
            filtered,
            x="age_name",
            y="display_value",
            color="location_name",
            barmode="group",

            error_y=(
                filtered["upper_display"]
                - filtered["display_value"]
            ).clip(lower=0),

            error_y_minus=(
                filtered["display_value"]
                - filtered["lower_display"]
            ).clip(lower=0),

            labels=labels
        )

    style_chart(fig, 350)

    fig.update_layout(
        xaxis_tickangle=-35,
        hovermode="x unified",
        legend=dict(orientation="h", y=-.35)
    )

    return fig


# =========================================================
# 15. IN-DASHBOARD RESEARCH REPORT
# =========================================================

@app.callback(
    Output("research-report", "children"),

    Input("uploaded-data", "data"),
    Input("disease-filter", "value"),
    Input("year-start-filter", "value"),
    Input("year-end-filter", "value"),
    Input("age-filter", "value"),
    Input("gender-filter", "value"),
    Input("metric-filter", "value"),
    Input("country-filter", "value"),
    Input("language-filter", "value"),
)
def update_research_report(
    uploaded_records, disease, start_year, end_year,
    ages, gender, metric, countries, lang
):
    data = filtered_report_data(
        uploaded_records,
        disease,
        start_year,
        end_year,
        ages,
        gender,
        metric,
        countries
    )

    return make_report_components(
        data,
        lang,
        disease,
        start_year,
        end_year,
        normalize_selection(ages),
        gender,
        metric,
        normalize_selection(countries)
    )


# =========================================================
# 16. PDF DOWNLOAD
# =========================================================

@app.callback(
    Output("download-report-pdf", "data"),

    Input("download-report-button", "n_clicks"),

    State("uploaded-data", "data"),
    State("disease-filter", "value"),
    State("year-start-filter", "value"),
    State("year-end-filter", "value"),
    State("age-filter", "value"),
    State("gender-filter", "value"),
    State("metric-filter", "value"),
    State("country-filter", "value"),
    State("language-filter", "value"),

    prevent_initial_call=True,
)
def download_pdf(
    n_clicks, uploaded_records, disease, start_year,
    end_year, ages, gender, metric, countries, lang
):
    if not n_clicks:
        return no_update

    data = filtered_report_data(
        uploaded_records,
        disease,
        start_year,
        end_year,
        ages,
        gender,
        metric,
        countries
    )

    pdf_bytes = create_pdf_report(
        data,
        lang,
        disease,
        start_year,
        end_year,
        normalize_selection(ages),
        gender,
        metric,
        normalize_selection(countries)
    )

    filename = (
        "ADHD_Research_Report_RU.pdf"
        if lang == "RU"
        else "ADHD_Research_Report_EN.pdf"
    )

    return dcc.send_bytes(pdf_bytes, filename)


# =========================================================
# 17. RUN APP
# =========================================================

if __name__ == "__main__":
    print("\nADHD RESEARCH DASHBOARD")
    print("Original records:", len(base_df))
    print("Conditions:", get_available_diseases(base_df))
    print("Countries:", len(COUNTRIES))
    print("Years:", YEARS)
    print("Open: http://127.0.0.1:8050")

    app.run(debug=True)