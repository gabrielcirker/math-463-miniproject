# MATH 463 Miniproject

T-cell, Treg, and IL-2 dynamics with ordinary differential equations and custom SINDy equation learning.

**[Project website](https://gabrielcirker.github.io/math-463-miniproject/)** · **[Final report](report/final-report.pdf)** · **[Notebook](immune_miniproject.ipynb)** · **[Python code](immune_miniproject.py)**

## Project files

| File or folder | Contents |
| --- | --- |
| [immune_miniproject.py](immune_miniproject.py) | Model, parameter experiments, SINDy, and numerical checks |
| [immune_miniproject.ipynb](immune_miniproject.ipynb) | The same workflow in Jupyter or Colab |
| [results/figures/](results/figures/) | Six figures in PNG and SVG formats |
| [results/data/](results/data/) | Simulated training, validation, and test trajectories |
| [results/](results/) | Learned equations, coefficients, metrics, and saved models |
| [paper_reference/](paper_reference/) | Digitized Figure 5B trace and calibration details |
| [START_HERE.txt](START_HERE.txt) | Setup and file guide |
| [MODEL_GUIDE.txt](MODEL_GUIDE.txt) | Equations, assumptions, and fitting method |
| [PROJECT_GUIDE.txt](PROJECT_GUIDE.txt) | Experiments and outputs |
| [RESULTS.txt](RESULTS.txt) | Explanation of the included run |
| [PAPER_COMPARISON.txt](PAPER_COMPARISON.txt) | Published baseline comparison and its limits |
| [report/final-report.pdf](report/final-report.pdf) | Final project report |

## Run the project

Use Python 3.10 or newer from the repository folder. A virtual environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python immune_miniproject.py --check-only
python immune_miniproject.py
python check_paper_trace.py
```

The notebook can also run in Jupyter or Colab. Run its cells in order. [requirements-tested.txt](requirements-tested.txt) records the package versions used for the supplied results; [run_manifest.json](results/run_manifest.json) records the run settings and output files. To preserve the supplied run, use `python immune_miniproject.py --output-dir results_new`.

## Model scope and sources

This is a three-state, constant-input interpretation of Eq. (13) in [Khailaie et al. (2013)](https://doi.org/10.3389/fimmu.2013.00474). The inputs are fixed at `k` and `lambda * k`; the two resting cell pools are omitted. It is not a complete five-state replication or a clinical model. The paper comparison uses a digitized plotted line, not the authors' numerical data.

SINDy method: [Brunton, Proctor, and Kutz (2016)](https://doi.org/10.1073/pnas.1517384113). This project uses its own implementation.

## Final report

[Read the report on the project website](https://gabrielcirker.github.io/math-463-miniproject/#report). The responsive reader embeds the original PDF and includes direct open and download links for browsers without embedded PDF support.

[Read the final report](report/final-report.pdf): *Modeling Immune Activation & Self-Tolerance: Conventional T-cell, Treg, and IL-2 Dynamics*, by Gabriel Cirker, Kennedy Jamison, and Cecilia Pirozzi (9-page PDF).

## Website maintenance

The site uses `index.html` and `assets/site.css`, with no build dependencies. GitHub Pages serves the root of `main`; published changes follow commits pushed to that branch. Scientific files retain their original relative paths. Two notebook output cells containing local machine paths were cleared for public sharing; source code and saved scientific results are unchanged.

