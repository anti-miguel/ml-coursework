# Machine-learning coursework

University coursework from the BSc in Data Science and Artificial Intelligence at UDIT. The notebooks retain their original Spanish text and result figures.

Resumen en español: prácticas de aprendizaje automático y neuronas artificiales, con documentación de métodos, resultados y disponibilidad de los materiales.

| Project | Goal and methods | Public material |
| --- | --- | --- |
| [Digital habits and wellbeing](digital-habits-wellbeing/) | Predict happiness from digital habits; exploratory analysis, Random Forest and tuned Gradient Boosting | Scripts, exploratory notebook and figures; the dataset is not included |
| [Single-neuron learning](deep-learning/) | Compare Perceptron, Adaline and sigmoid neurons on AND, OR and XOR | Two runnable notebooks, including the restored backup, with 12 retained result figures |

## Run the public notebooks

Use Python 3.13 (the version used for verification). From the repository root:

```sh
python -m venv .venv
```

Activate `.venv` using your shell's activation command, then install and launch:

```sh
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open either notebook in `deep-learning/` with the Python 3 kernel and run its cells in order. No dataset, credentials, network access or external storage is needed by these notebooks.

For a headless execution check with behavioural assertions:

```sh
python tools/check_notebooks.py
```

## Material boundaries

The wellbeing project's final report was written with a classmate and is not included; the code here is the analysis behind it. Course reports, datasets and teaching material are not published.

No dataset is distributed: the wellbeing CSV came without a verifiable redistribution licence. Obtain it separately and pass its path with `--data` (see the project README).

See [VALIDATION.md](VALIDATION.md) for execution results.
