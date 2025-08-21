# handsOnDE_yt_api

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

## Overview

**handsOnDE_yt_api** is a hands-on exploration project focused on building a YouTube API-based ETL (Extract, Transform, Load) data pipeline. It is designed as a learning platform to practice modern data engineering workflows, including basic ETL, dashboarding, and plans for advanced analytics and orchestration.

- **Repository:** [GentleClash/handsOnDE_yt_api](https://github.com/GentleClash/handsOnDE_yt_api)
- **Primary Language:** Jupyter Notebook (Python ecosystem)
- **License:** GNU General Public License v3.0

## Features

- **Extract:** Pull data from the YouTube API.
- **Transform:** Clean and reshape raw YouTube data for downstream use.
- **Basic Load:** Save transformed data to local or cloud storage.
- **Dashboard:** Visualize YouTube metrics; engagement rate dashboard is in progress.
- **Planned Enhancements:**
  - Time Series Analysis
  - Categorical Analysis
  - Channel Analysis
  - ORM Integration
  - Scheduled Jobs (Cron)
  - Airflow Pipeline Orchestration

## Project Structure

```
.
├── dashboard.py          # Dashboard and visualization logic
├── exploration/          # Jupyter Notebooks and data exploration scripts
├── main.py               # Main pipeline entry point
├── requirements.txt      # Python dependencies
├── poetry.lock           # Poetry lock file for reproducible environments
├── pyproject.toml        # Poetry project configuration
├── src/                  # Source code modules
├── LICENSE
├── README.md
└── .gitignore
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/GentleClash/handsOnDE_yt_api.git
cd handsOnDE_yt_api
```

### 2. Install Dependencies

You can use either `pip` or `poetry`:

**With pip:**
```bash
pip install -r requirements.txt
```

**With poetry:**
```bash
poetry install
```

### 3. Run the Pipeline

```bash
python main.py
```

### 4. Explore & Visualize

- Use `dashboard.py` to launch dashboards or generate visual analytics.
- Check the `exploration/` directory for Jupyter notebooks and exploratory scripts.

## Roadmap

- [x] Extract
- [x] Transform
- [x] Basic Load
- [x] Dashboard
- [x] Containerization
- [ ] Time Series Analysis
- [ ] Categorical Analysis
- [ ] Channel Analysis
- [ ] ORM Integration
- [ ] Cron Jobs
- [ ] Airflow Orchestration

## Contributing

Contributions, issues, and feature requests are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the terms of the [GNU GPLv3](LICENSE).

---
*Part of a series of personal projects to build hands-on data engineering skills in ETL and ELT workflows.*
