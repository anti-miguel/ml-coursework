# Perceptrón, Adaline y neurona sigmoide

## Objetivo y datos

Comparar el aprendizaje de las compuertas AND, OR y XOR con tres modelos de una sola neurona. Las cuatro entradas binarias y sus etiquetas se definen en los cuadernos; no se requieren datos externos ni LMDB.

## Cuadernos y método

- `entrega.ipynb`: entrega con Perceptrón (hasta 50 épocas), Adaline (100 épocas), comparación de tasas de aprendizaje y neurona sigmoide (500 épocas).
- `pregunta1_perceptron.ipynb`: copia restaurada del archivo de respaldo. Incluye Adaline durante 400 épocas y registra la primera época con las cuatro clasificaciones correctas.

El Perceptrón actualiza los pesos por ejemplo. Adaline utiliza descenso de gradiente por lotes y un umbral de 0.5. La neurona sigmoide compara error cuadrático y entropía cruzada. Se conservan las seis figuras de cada cuaderno.

## Resultados registrados y comprobados

El Perceptrón aprende AND y OR en 4 épocas y no resuelve XOR en 50 épocas. En el cuaderno restaurado, Adaline clasifica las cuatro entradas de AND por primera vez en la época 192 y las de OR en la 65; los costes finales son 0.042306 y 0.034506. XOR termina con un coste de 0.127314 y no clasifica correctamente las cuatro entradas.

Con la sigmoide, entropía cruzada alcanza las cuatro clasificaciones correctas en las épocas 34 (AND) y 40 (OR), frente a 109 y 121 con MSE. Para XOR, las salidas permanecen en 0.5 con la inicialización utilizada; los costes son aproximadamente 0.125 (MSE) y 0.693147 (CE). Las escalas de ambas pérdidas son diferentes.

Las versiones no son idénticas: los resultados de Adaline con 400 épocas no describen la ejecución de 100 épocas de `entrega.ipynb`. El PDF original contiene cifras y etiquetas inconsistentes con los cuadernos; permanece en `_private/` y no se usa como referencia de ejecución.

## Ejecución

Instalar `requirements.txt` desde la raíz del repositorio y abrir los cuadernos con JupyterLab. Ejecutar las celdas en orden con el kernel Python 3. Son entregas en formato notebook, sin interfaz de línea de comandos. Para comprobar ambos cuadernos sin interfaz gráfica:

```sh
python tools/check_notebooks.py
```

El comando anterior se ejecuta desde la raíz. Comprueba la ejecución completa, el aprendizaje de AND/OR y la limitación de representación en XOR. Los originales se conservaron sin modificaciones; las copias públicas eliminan metadatos específicos del entorno.
