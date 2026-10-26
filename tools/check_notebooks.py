"""Execute the standalone neuron notebooks and check learning behaviour."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import json
import os
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "_private" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


def check(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    try:
        with redirect_stdout(StringIO()), patch.object(plt, "show"):
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    exec(
                        compile("".join(cell["source"]), f"{path.name}:cell-{index}", "exec"),
                        namespace,
                    )

        results = namespace.get("resultados_p1", namespace.get("resultados"))
        for gate in ("AND", "OR"):
            np.testing.assert_array_equal(
                results[gate]["predicciones"], namespace["compuertas"][gate]
            )
        assert not np.array_equal(
            results["XOR"]["predicciones"], namespace["compuertas"]["XOR"]
        )
        for result in namespace["resultados_p3"]:
            assert np.isfinite(result["historial"]).all()
            if result["compuerta"] == "XOR":
                assert not result["clasifica_bien"]
                np.testing.assert_allclose(result["salidas"], 0.5)
            else:
                assert result["clasifica_bien"]
        print(f"PASS {path.name}: all cells executed; AND/OR learned; XOR remains unsolved")
    finally:
        plt.close("all")


if __name__ == "__main__":
    for filename in ("entrega.ipynb", "pregunta1_perceptron.ipynb"):
        check(ROOT / "deep-learning" / filename)
