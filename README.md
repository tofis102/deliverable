# Code for AI-led Qualitative Interviews and Focus Groups (with AI and Human Participants)

## Repository Summary

This repository enables you to perform semi-structured:

1. ...qualitative interviews with a human participant using an AI-powered interviewer.
2. ...focus group discussions led by an AI-powered moderator, with several well-prompted AI agents simulating participants, and:
   - ...__no__ human participant.
   - ...__one__ human participant.

### Option 1: AI Interview Replication

Option 1 replicates two recent papers – Chopra & Haaland (2023) and Geiecke & Jaravel (2024) – which introduced this methodology to economic science. This deliverable extends Geiecke & Jaravel (2024) by adding the option to run qualitative AI interviews using a local AI model (e.g., Ollama `gemma3`) and provides an easy way to deploy the application.

### Option 2: AI-Powered Focus Group

Option 2 is the main contribution, extending the AI interview to a focus group. Local AI models can be used, but for complex prompts, larger models like GPT-4o or GPT-5 via OpenAI are recommended for stronger reasoning.

### Further Information Is Provided via the Notebook

**This `README.md` is a pre-stage guide to `Notebook.qmd`, which situates the extension in the literature, provides step-by-step instructions for interviews and focus groups, explains coding decisions, shares limitations, explores usage and extensions, and analyzes AI focus groups qualitatively.**

## Clone GitHub Repository & Environment Setup

### Step 1: Clone Repository

You can clone this GitHub repository by executing the following code. If you want to save the repository at the specific location, then set the specific path to
```bash
cd "path_project"
```
first and run
```bash
git clone https://github.com/tofis102/deliverable.git
cd deliverable
```

### Step 2: Setup the Local Environment to Render the Notebook

To be able to run the core document `Notebook.qmd` of this deliverable you have to execute the following code. This code creates a virtual environment, e.g. named .venv_notebook, to install necessary packages in a clean environment, avoiding clashing dependencies, and activates the enviornment. In your command-line terminal run:

**Option A: Python Virtual Environment (PowerShell)**
```powershell
python -m venv .venv_notebook
.\.venv_notebook\Scripts\activate
pip install -r requirements_notebook.txt
```

**Option B: Conda Environment (Anaconda Prompt/PowerShell)**

If you have not installed [miniconda](https://docs.anaconda.com/miniconda/miniconda-install/) yet, you have to download it first.
```powershell
conda env create -f environment_notebook.yml
conda activate .venv_notebook
```

### Step 3: Render & Preview Notebook
To render and preview the Notebook `Notebook.qmd` yourself, install [Quarto](https://quarto.org/docs/get-started/)
```powershell
quarto render Notebook.qmd
quarto preview Notebook.qmd
```
Alternatively you can use the shortcut `Crtl`+`Shift`+`V` if you work in VS Code with the Quarto extension.

## References
This repository builds largely on previous works by Chopra & Haaland (2023) and Geiecke & Jaravel (2024). 

For Chopra & Haaland (2023) refer either to the [repository](https://github.com/fchop/interviews) or 
```
Chopra, Felix and Haaland, Ingar, Conducting Qualitative Interviews with AI (2023). CESifo Working Paper No. 10666, Available at SSRN: https://ssrn.com/abstract=4583756 or http://dx.doi.org/10.2139/ssrn.4583756
```
You can use the suggested Bibtex entry:
```bibtex
@article{ChopraHaaland2023,
  title={Conducting Qualitative Interviews with AI},
  author={Chopra, Felix and Haaland, Ingar},
  journal={CESifo Working Paper No. 10666},
  url={https://ssrn.com/abstract=4583756},
  year={2023}
}
```

For Geiecke & Jaravel (2024) refer either to the [repository](https://github.com/friedrichgeiecke/interviews) or 
```
Geiecke, Friedrich and Jaravel, Xavier, Conversations at Scale: Robust AI-led Interviews with a Simple Open-Source Platform (2024). Available at SSRN: https://ssrn.com/abstract=4974382 
```
You can use the suggested Bibtex entry:
```bibtex
@article{geieckejaravel2024,
  title={Conversations at Scale: Robust AI-led Interviews with a Simple Open-Source Platform},
  author={Geiecke, Friedrich and Jaravel, Xavier},
  url={https://ssrn.com/abstract=4974382},
  year={2024}
}
```

**Note:** There exist also updated versions of the papers on the authors' websites. The most recent version of Chopra & Haaland is from September 2024. The most recent version of Geiecke & Jaravel is from July 2025.



