# Proyecto final: salud mental, redes sociales y modelos predictivos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Carga de datos

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Análisis de hábitos digitales y bienestar")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data" / "Mental_Health_and_Social_Media_Balance_Dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "resultados_proyecto")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error("No se encuentra el CSV. Consulta README.md y utiliza --data.")

    DATA_PATH = args.data
    df = pd.read_csv(DATA_PATH)

    numerical_cols = [
        'Age', 'Daily_Screen_Time(hrs)', 'Sleep_Quality(1-10)',
        'Stress_Level(1-10)', 'Days_Without_Social_Media',
        'Exercise_Frequency(week)', 'Happiness_Index(1-10)'
    ]
    features_num = [c for c in numerical_cols if c != 'Happiness_Index(1-10)']
    cat_cols = ['Gender', 'Social_Media_Platform']
    target = 'Happiness_Index(1-10)'

    # 2. Revision inicial
    print(df.shape)
    print(df.head())
    print(df.dtypes)
    print(df.isnull().sum())
    print(df.describe())

    # 3. Analisis exploratorio basico
    print('Distribucion de genero:')
    print(df['Gender'].value_counts())
    print('Promedio de felicidad:', df[target].mean())
    print('Screentime medio:', df['Daily_Screen_Time(hrs)'].mean())
    print('Correlaciones con felicidad:')
    print(df[numerical_cols].corr()[target].sort_values(ascending=False))

    plt.figure(figsize=(10, 7))
    mask = np.triu(np.ones_like(df[numerical_cols].corr(), dtype=bool))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0, mask=mask, fmt='.2f')
    plt.title('Matriz de correlacion')
    plt.tight_layout()
    plt.show()

    for col in ['Stress_Level(1-10)', 'Daily_Screen_Time(hrs)', 'Sleep_Quality(1-10)']:
        plt.figure(figsize=(6, 4))
        sns.regplot(data=df, x=col, y=target, scatter_kws={'alpha': 0.5})
        plt.title(f'{col} vs {target} | r = {df[col].corr(df[target]):.2f}')
        plt.tight_layout()
        plt.show()

    # 4. Random Forest y Gradient Boosting basicos
    X = df[features_num]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_rmse = mean_squared_error(y_test, rf_pred) ** 0.5
    rf_r2 = r2_score(y_test, rf_pred)

    basic_gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
    basic_gb.fit(X_train_scaled, y_train)
    gb_pred = basic_gb.predict(X_test_scaled)
    gb_rmse = mean_squared_error(y_test, gb_pred) ** 0.5
    gb_r2 = r2_score(y_test, gb_pred)

    print('Random Forest:', rf_rmse, rf_r2)
    print('Gradient Boosting basico:', gb_rmse, gb_r2)

    # 5. Gradient Boosting con categoricas y ajuste de hiperparametros
    X_all = df[features_num + cat_cols]
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), features_num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', GradientBoostingRegressor(random_state=42))
    ])

    param_grid = {
        'model__n_estimators': [100, 200, 300, 500],
        'model__learning_rate': [0.01, 0.03, 0.05, 0.1],
        'model__max_depth': [2, 3, 4],
        'model__subsample': [0.8, 1.0]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid.fit(X_train_all, y_train_all)

    best_model = grid.best_estimator_
    best_pred = best_model.predict(X_test_all)
    best_rmse = mean_squared_error(y_test_all, best_pred) ** 0.5
    best_r2 = r2_score(y_test_all, best_pred)

    print('Mejores hiperparametros:', grid.best_params_)
    print('Gradient Boosting optimizado:')
    print('RMSE:', best_rmse)
    print('R2:', best_r2)

    # 6. Graficos de modelo
    comparison = pd.DataFrame({
        'Modelo': ['Random Forest', 'Gradient Boosting basico', 'Gradient Boosting optimizado'],
        'RMSE': [rf_rmse, gb_rmse, best_rmse],
        'R2': [rf_r2, gb_r2, best_r2]
    })
    print(comparison)

    sns.barplot(data=comparison, x='Modelo', y='R2')
    plt.title('Comparacion de R2')
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.show()

    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    importances = best_model.named_steps['model'].feature_importances_
    importance_df = pd.DataFrame({'Variable': feature_names, 'Importancia': importances}).sort_values('Importancia', ascending=False)
    print(importance_df.head(10))


if __name__ == "__main__":
    main()
