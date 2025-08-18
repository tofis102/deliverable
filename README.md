# Code for AI-Led Qualitative Interviews and Focus Groups (with AI and Human Participants)

## Repository Summary
This repository enables you to perform a semi-structured...

1. ...qualitative interview with a human participant by using an AI-powered interviewer to conduct the sessions.

2. ...focus group discussion led by an AI-powered moderator, several well-prompted AI agents simulating participants, and:

   - ...__no__ human participant.
   - ...__one__ human participant.

### Option 1: AI Interview Replication

Option 1 mainly replicates two recent working papers – Chopra & Haaland (2023) and Geiecke & Jaravel (2024) – which introduced this methodology to the field of economic science. This deliverable extends Geiecke & Jaravel (2024) by adding the option to run the qualitative AI interview using a local AI model (e.g., Ollama `gemma3`) and provides an easy way to deploy the application.

### Option 2: AI-Powered Focus Group

Option 2 constitutes the major work of the student for this deliverable, extending the AI interview to an AI-powered focus group. This part of the deliverable can also be run using local AI models. However, due to the prompts' complexity and length, using well-developed larger closed-source models such as GPT-4o or GPT-5 powered by OpenAI is recommended. These models offer much stronger reasoning capabilities than smaller local models.

### Further Information Is Provided via the Notebook
**This `README.md` serves as a pre-stage guide to the `Notebook.qmd` file, which among other things briefly situates the extension within the literature, instructs how to setup all interviews and focus group discussions step by step, explains the code and coding decisions, shares limitations encountered, explores potential usage and extensions, analyses the AI focus group discussion qualitatively, and compares them to real human focus group discussions.**

## Clone GitHub Repository & Environment Setup to Render Notebook
**Step 1: Clone Repository**
You can clone this GitHub repository by executing the following code. If you want to save the repository at the specific location, then set the specific path to
```bash
cd "path_project"
```
first and run
```bash
git clone https://github.com/tofis102/deliverable.git
cd deliverable
```

**Step 2: Setup the Local Environment to Render the Notebook** 
To be able to run the core document `Notebook.qmd` of this deliverable yo have to execute the following code.
Create a virtual environment, e.g. named .venv_notebook, and activate it, so as to install necessary packages in a clean environment, guaranteeing no clashing dependencies. In your command-line terminal run:
```powershell
python -m venv .venv_notebook
.\.venv_notebook\Scripts\activate
pip install -r requirements_notebook.txt
```
__Alternatively__, you can use Anaconda Prompt or equivalent supporting terminals to create and activate the local environment
```powershell
conda env create -f environment_notebook.yml
conda activate .venv_notebook
```

**Step 3: Render & Preview Notebook**
To render and preview the Notebook `Notebook.qmd` yourself, install [Quarto](https://quarto.org/docs/get-started/)
```bash
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
```
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
```
@article{geieckejaravel2024,
  title={Conversations at Scale: Robust AI-led Interviews with a Simple Open-Source Platform},
  author={Geiecke, Friedrich and Jaravel, Xavier},
  url={https://ssrn.com/abstract=4974382},
  year={2024}
}
```

**Note:** There exist also updated versions of the papers on the authors' websites. The most recent version of Chopra & Haaland is from September 2024. The most recent version of Geiecke & Jaravel is from July 2025.



