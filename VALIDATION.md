# Validation

Verified with Python 3.13.2 and the library versions in `requirements.txt`.

| Material | Check | Outcome |
| --- | --- | --- |
| `deep-learning/entrega.ipynb` | Execute all three code cells, then assert learned AND/OR labels and unsolved XOR | Passed |
| `deep-learning/pregunta1_perceptron.ipynb` | Execute all three code cells; assert the same behaviour and finite sigmoid loss histories | Passed |
| `digital-habits-wellbeing/codigo_proyecto_redes_salud_mental.py` | Execute against a locally obtained copy of the dataset | RMSE 0.936278; R² 0.630991 |
| `digital-habits-wellbeing/codigo_proyecto_final_salud_mental.py` | Execute the complete five-fold, 192-combination search | Tuned Gradient Boosting RMSE 0.903383; R² 0.656466 |
| `digital-habits-wellbeing/analisis_exploratorio.ipynb` | Execute both code cells with the relative data path | Passed; RF RMSE 0.936 and R² 0.631 |

Notebook code was executed in order through Python with a headless plotting backend; JupyterLab itself was not launched. Figures were visually inspected. Results come from one fixed train/test split and carry no uncertainty intervals.

Not published: the course reports, the datasets, and teaching material whose redistribution rights are unclear.
