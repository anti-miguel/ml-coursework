"""
Proyecto final - Técnicas de Procesamiento de Datos
Análisis de redes sociales, salud mental e índice de felicidad.

Requisitos:
    pip install pandas numpy matplotlib seaborn scipy scikit-learn

Uso:
    python codigo_proyecto_redes_salud_mental.py

El script espera el archivo:
    Mental_Health_and_Social_Media_Balance_Dataset.csv
en la misma carpeta del script.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Análisis de hábitos digitales y bienestar")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data" / "Mental_Health_and_Social_Media_Balance_Dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "resultados_proyecto")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error("No se encuentra el CSV. Consulta README.md y utiliza --data.")

    DATA_PATH = args.data
    OUT_DIR = args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("ANÁLISIS DE REDES SOCIALES, SALUD MENTAL Y FELICIDAD")
    print("=" * 60)

    print("\n1. Información básica")
    print(f"Forma del dataset: {df.shape}")
    print("\nTipos de datos:")
    print(df.dtypes)
    print("\nValores faltantes:")
    print(df.isnull().sum())
    print("\nEstadísticas descriptivas:")
    print(df.describe())

    numerical_cols = [
        "Age",
        "Daily_Screen_Time(hrs)",
        "Sleep_Quality(1-10)",
        "Stress_Level(1-10)",
        "Days_Without_Social_Media",
        "Exercise_Frequency(week)",
    ]
    target = "Happiness_Index(1-10)"

    print("\n2. Distribución de variables categóricas")
    print("\nGénero:")
    print(df["Gender"].value_counts())
    print("\nPlataformas:")
    print(df["Social_Media_Platform"].value_counts())

    print("\n3. Correlaciones con felicidad")
    corr = df[numerical_cols + [target]].corr()[target].sort_values(ascending=False)
    print(corr)

    # Matriz de correlación
    plt.figure(figsize=(10, 7))
    corr_matrix = df[numerical_cols + [target]].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, mask=mask, fmt=".2f")
    plt.title("Matriz de correlación de variables numéricas")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "matriz_correlacion.png", dpi=200)
    plt.close()

    print("\n4. Análisis por plataforma")
    platform_analysis = df.groupby("Social_Media_Platform").agg({
        target: "mean",
        "Daily_Screen_Time(hrs)": "mean",
        "Stress_Level(1-10)": "mean",
        "Sleep_Quality(1-10)": "mean",
        "User_ID": "count",
    }).round(2)
    platform_analysis = platform_analysis.rename(columns={"User_ID": "User_Count"})
    print(platform_analysis)

    platform_analysis.to_csv(OUT_DIR / "analisis_plataformas.csv")

    # Detección visual de outliers
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df[numerical_cols + [target]])
    plt.title("Diagrama de cajas de variables numéricas")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "boxplot_outliers.png", dpi=200)
    plt.close()

    # Grupos de edad
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[15, 25, 35, 50],
        labels=["Joven (15-25)", "Adulto (26-35)", "Edad Media (36-50)"],
    )
    age_group_analysis = df.groupby("Age_Group").agg({
        target: "mean",
        "Daily_Screen_Time(hrs)": "mean",
        "Stress_Level(1-10)": "mean",
        "Sleep_Quality(1-10)": "mean",
        "User_ID": "count",
    }).round(2)
    age_group_analysis = age_group_analysis.rename(columns={"User_ID": "User_Count"})
    print("\n5. Análisis por grupo de edad")
    print(age_group_analysis)
    age_group_analysis.to_csv(OUT_DIR / "analisis_grupos_edad.csv")

    # Modelo Random Forest
    print("\n6. Modelo de aprendizaje automático: Random Forest Regressor")
    X = df[numerical_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")

    importance_df = pd.DataFrame({
        "Variable": numerical_cols,
        "Importancia": model.feature_importances_,
    }).sort_values("Importancia", ascending=False)

    print("\nImportancia de variables:")
    print(importance_df)
    importance_df.to_csv(OUT_DIR / "importancia_variables.csv", index=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=importance_df, x="Importancia", y="Variable")
    plt.title("Importancia de las variables en el modelo de predicción de felicidad")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "importancia_variables.png", dpi=200)
    plt.close()

    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN DE HALLAZGOS CLAVE")
    print("=" * 60)
    print(f"Usuarios: {len(df)}")
    print(f"Plataformas: {df['Social_Media_Platform'].nunique()}")
    print(f"Felicidad media: {df[target].mean():.2f}")
    print(f"Screentime medio diario: {df['Daily_Screen_Time(hrs)'].mean():.2f} horas")
    print(f"Mayor correlación positiva con felicidad: {corr.index[1]} ({corr.iloc[1]:.3f})")
    print(f"Mayor correlación negativa con felicidad: {corr.index[-1]} ({corr.iloc[-1]:.3f})")
    print(f"Plataforma con mayor felicidad: {platform_analysis[target].idxmax()} ({platform_analysis[target].max():.2f})")
    print(
        "Plataforma con mayor screentime: "
        f"{platform_analysis['Daily_Screen_Time(hrs)'].idxmax()} "
        f"({platform_analysis['Daily_Screen_Time(hrs)'].max():.2f} horas)"
    )


if __name__ == "__main__":
    main()
