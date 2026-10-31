---

<div align="center">

[![Sourcery](https://img.shields.io/badge/Sourcery-enabled-brightgreen)](https://sourcery.ai)
[![Poetry](https://img.shields.io/badge/packaging-poetry-cyan.svg)](https://python-poetry.org/)
[![GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

</div>

<hr>

# Sigma Chatbot

A university information chatbot built with Streamlit, OpenAI, LangChain, ChromaDB, and Google APIs.

This repository is a fork of [TOBB-GPT](https://github.com/TOBB-ETU-CS-Community/TOBB-GPT), originally developed for TOBB University of Economics and Technology.


[Upstream TOBB-GPT demo](https://tobb-gpt.streamlit.app/)


## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before setting up the project, ensure that you have the following prerequisites installed on your machine:

- Python (version 3.10.10):
  (https://www.python.org/downloads/)
- Poetry (version 1.4.0):
  (https://python-poetry.org/docs/#installation)

## Installation

1. Clone the project repository:

```
git clone https://github.com/ataturhan21/Sigma-Chatbot.git
```

2. Navigate to the project directory:

```
cd Sigma-Chatbot
```

3. Install project dependencies using Poetry:

```
poetry install --no-root
```

This installs the locked dependencies without trying to package the Streamlit application.

## Configuration

1. Create `.streamlit/secrets.toml` in the repository root, using [`secrets.example.toml`](secrets.example.toml) as a template.
2. Set `GOOGLE_CSE_ID` to your Google Programmable Search Engine ID and `GOOGLE_API_KEY` to its API key. The application reads both when it starts.
3. Select a model in the sidebar and enter the matching OpenAI or Hugging Face credential. The input is masked.

The current UI uses web search to collect pages for retrieval. The code also contains an alternative document-index workflow based on [`static/Links.xlsx`](static/Links.xlsx); it is not the active UI path.

Run the application from the repository root so that `static/` assets and local Chroma paths resolve. The existing `.gitignore` excludes `.streamlit/`; keep real credentials in that ignored directory.

These are the original LangChain, OpenAI, and Chroma integrations recorded in `pyproject.toml` and `poetry.lock`. Provider models, access permissions, and API compatibility must be checked before a live deployment.

## Usage

1. Activate the virtual environment created by Poetry:

```
poetry shell
```

2. Run the project:

```
streamlit run "tobb_gpt/app.py"
```

## Contributing

If you would like to contribute to this project, follow the guidelines below:

1. Fork the repository and clone it locally.

2. Create a new branch for your feature or bug fix:
   ```
   git checkout -b feature/your-feature
   ```

3. Make your changes and commit them with descriptive commit messages.

4. Push your changes to your forked repository:
   ```
   git push origin feature/your-feature
   ```

5. Open a pull request in the original repository, describing your changes in detail.


## License

GPL v3

---
