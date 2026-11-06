# ADHD Research Dashboard

An interactive, bilingual (English/Russian) research dashboard for
exploring the global prevalence, demographic patterns, geographic
distribution, and historical trends of Attention-Deficit/Hyperactivity
Disorder (ADHD).

The project combines Global Burden of Disease (GBD) estimates for
2019--2023 with selected external research and public-health sources to
provide a concise, evidence-based overview. It includes interactive
visualizations and PDF report export.

## Key features

-   **Bilingual interface:** switch between English and Russian.
-   **Global prevalence:** explore estimated ADHD prevalence and case
    counts.
-   **Age distribution:** compare prevalence across age groups.
-   **Age at first diagnosis:** view findings from published
    population-based research.
-   **Sex differences:** compare estimates for male and female
    populations.
-   **ADHD presentations:** summarize evidence on inattentive,
    hyperactive-impulsive, and combined presentations.
-   **ADHD and ASD comorbidity:** present findings from a systematic
    review, with methodological cautions.
-   **Geographic distribution:** compare estimates across the countries
    included in the supplied dataset.
-   **Historical trends:** explore changes in estimated prevalence from
    2019 to 2023, alongside selected external evidence on recorded
    diagnoses.
-   **PDF export:** generate a research report with charts, key
    findings, and source links.

## Research scope and data

The dashboard focuses on ADHD. Autism spectrum disorder (ASD) is
included only in the context of ADHD co-occurrence; this is not a
separate ASD prevalence dashboard.

### Included datasets

Place these three CSV files in the `data/` directory:

``` text
data/
├── 2019-23.csv
├── all ages_19-23.csv
└── geo_distribution.csv
```

The supplied GBD dataset covers **2019--2023**. It includes global
age/sex estimates and geographic estimates for the countries represented
in `geo_distribution.csv`.

The dashboard also references selected external publications, including
research on age at diagnosis, ADHD presentations, ADHD--ASD
co-occurrence, and recorded diagnosis rates. Publication year and
data-collection year should not be treated as interchangeable.

## Important interpretation notes

-   **Prevalence estimates are not the same as diagnosed-case counts.**
    GBD values are modelled population estimates, not a count of people
    with a confirmed clinical diagnosis.
-   **The GBD files do not identify individual diagnoses**, age at first
    diagnosis, ADHD presentation, or whether a person has both ADHD and
    ASD.
-   **Diagnosis rates require separate evidence.** Survey or
    healthcare-record studies are used for recorded diagnosis trends;
    these should not be presented as if they came from the GBD
    prevalence CSVs.
-   **Diagnostic coverage gaps are estimates, not direct counts of
    undiagnosed people.** Where an external study compares recorded
    diagnoses with epidemiological reference estimates, the difference
    should be interpreted according to that study's methodology.
-   **Geographic comparisons are limited to the countries in the
    supplied file.** They should not be described as a complete ranking
    of every country worldwide.
-   **Male/Female categories in the GBD data refer to the dataset's sex
    categories**, not gender identity.
-   **Uncertainty intervals matter.** The `lower` and `upper` fields
    represent the lower and upper bounds of the 95% uncertainty interval
    where provided.
-   Results from different studies may use different populations, age
    ranges, screening tools, and diagnostic criteria. They should not be
    combined as if they measured exactly the same thing.

## Project structure

``` text
ADHD_Research/
├── app.py
└── data/
    ├── 2019-23.csv
    ├── all ages_19-23.csv
    └── geo_distribution.csv
```

## Installation

Python 3.10 or newer is recommended.

1.  Clone the repository:

    ``` bash
    git clone <YOUR_REPOSITORY_URL>
    cd ADHD_Research
    ```

2.  (Recommended) Create and activate a virtual environment:

    **macOS / Linux**

    ``` bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

    **Windows**

    ``` bash
    python -m venv .venv
    .venv\Scripts\activate
    ```

3.  Install the dependencies:

    ``` bash
    pip install dash pandas plotly reportlab
    ```

## Run the dashboard

From the project root, run:

``` bash
python app.py
```

Open the local address printed in the terminal (typically
`http://127.0.0.1:8050/`) in your browser.

If the application reports that a CSV file is missing, check that all
three required files are present in the `data/` directory and that their
filenames match the names listed above.

## Data sources and further reading

The dashboard uses the supplied GBD CSV extracts and selected external
research. Key references include:

-   **Global Burden of Disease (GBD) results:** source of the modelled
    prevalence estimates included in the CSV extracts. The data files
    supplied with this project define the exact years, locations, age
    groups, and measures available in the dashboard.
-   **Age at first recorded diagnosis in Wales:**
    https://pubmed.ncbi.nlm.nih.gov/38864317/
-   **ADHD diagnosis trends among children and adolescents in the United
    States (CDC, 2019--2025):**
    https://www.cdc.gov/mmwr/volumes/75/wr/mm7542a2.htm
-   **Recorded ADHD in England, 2000--2025:**
    https://pmc.ncbi.nlm.nih.gov/articles/PMC13453955/
-   **ADHD and ASD co-occurrence systematic review (2024):**
    https://link.springer.com/article/10.1007/s40489-024-00443-4
-   **Meta-analysis of ADHD presentations in children:**
    https://pmc.ncbi.nlm.nih.gov/articles/PMC10120242/

Always consult the original publications for study populations,
definitions, methods, and limitations before reusing a statistic.

## Intended use

This project is intended for educational and research-exploration
purposes. It is not a diagnostic tool and must not be used to diagnose
ADHD or ASD, make individual clinical decisions, or replace professional
medical advice.

## Current data coverage

  Component                     Coverage
  ----------------------------- ----------------------------------------------
  GBD prevalence data           2019--2023
  Global age and sex patterns   Available in supplied GBD extracts
  Geographic estimates          Countries included in `geo_distribution.csv`
  External diagnosis research   Varies by source and study period
  ADHD--ASD co-occurrence       Evidence summarized from selected studies

## Notes for contributors

-   Keep the original CSV column names unless the data-loading and
    preprocessing code is updated accordingly.
-   Do not interpret modelled prevalence as confirmed diagnoses.
-   When adding a new source, record its publication year, study/data
    period, population, measure, and URL.
-   Preserve the distinction between prevalence, incidence, recorded
    diagnosis, and diagnostic coverage.
