# FinGPT Research Assistant

![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-educational-green)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Example Outputs](#example-outputs)
- [Notes](#notes)
- [License](#license)


---

## Project Overview
FinGPT is a Python-based autonomous financial research platform that integrates **structured and unstructured data sources** to generate investment insights. The system combines **fundamental analysis**, **macroeconomic indicators**, and **news sentiment** with an optional **LLM-based recommendation engine** to assist in evaluating stock opportunities efficiently.

---

## Features

- **Fundamental Analysis**: Evaluates Earnings Growth, Valuation, Momentum, Stability, Analyst Sentiment, Sector Health, and Company Maturity.  
- **Macroeconomic Analysis**: Pulls data from FRED to compute unemployment, CPI, and interest rate indicators.  
- **News Sentiment**: Uses MPNet embeddings and a financial PhraseBank classifier to extract sentiment from news articles.  
- **LLM Integration**: Optional module synthesizes insights from structured data and news to provide natural-language investment recommendations.  
- **Modular Design**: Python project organized into `analysis`, `data`, `models`, and `utils` for easy maintainability and extension.  

---

## Project Structure
```
FinGPT Research Assistant/
├── analysis/ # Modules for fundamentals, sentiment, macro, score calculation
├── data/ # Handlers for news, FRED, Yahoo, and PhraseBank loader
├── models/ # ML models, embeddings, trained classifiers
├── utils/ # Helper functions and logging setup
├── main.py # Main entry point for running the analysis
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Ignored files and folders
```


---

## Installation

1. **Clone the repository**:

```
git clone https://github.com/Verdent06/FinGPT.git
cd "FinGPT Research Assistant"
```
2. **Create a virtual environment**:

```
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
```

3. **Install dependencies**:

```
pip install -r requirements.txt
```
---
## Usage
1. **Run the main analysis script**:

```
python main.py
```
2. **Modify configuration files in config/ for custom settings**.

3. **Outputs include**:

- Fundamental scores

- Macroeconomic analysis

- News sentiment summaries

- Optional LLM-generated recommendation

---
## Notes
Large files such as models or reference data are ignored in Git and stored locally.

Recommended Python version: 3.9+

---
## License
This project is intended for personal and educational purposes only.
For commercial usage, please contact the author.
