# -*- coding: utf-8 -*-
"""
utils_modelo.py
Funciones reutilizables para el análisis de elasticidad-precio del
dataset "Pizza Sales Dataset" (Kaggle: nextmillionaire/pizza-sales-dataset).

Este módulo es importado tanto por el notebook exploratorio (01_modelo_elasticidad_precio.ipynb)
como por la app de Streamlit (app_streamlit.py), siguiendo la lógica de la guía:
notebook -> refactorizar en funciones -> reutilizar en Streamlit.

Columnas esperadas del CSV original (schema de nextmillionaire/pizza-sales-dataset):
    pizza_id, order_id, pizza_name_id, quantity, order_date, order_time,
    unit_price, total_price, pizza_size, pizza_category, pizza_ingredients, pizza_name
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# ---------------------------------------------------------------------------
# 1. Carga y limpieza
# ---------------------------------------------------------------------------
def cargar_datos(ruta_csv: str) -> pd.DataFrame:
    """Carga el CSV crudo y aplica limpieza mínima."""
    df = pd.read_csv(ruta_csv)

    # Normalizar nombres de columnas por si vienen con mayúsculas/espacios
    df.columns = [c.strip().lower() for c in df.columns]

    # Parseo de fecha (algunos exports traen formato dd/mm/yyyy)
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        df["dia_semana"] = df["order_date"].dt.day_name()
        df["mes"] = df["order_date"].dt.month

    # Quitar filas sin precio o cantidad válidos
    df = df.dropna(subset=["unit_price", "quantity"])
    df = df[(df["unit_price"] > 0) & (df["quantity"] > 0)]

    return df


# ---------------------------------------------------------------------------
# 2. Agregación a nivel de producto (necesaria porque el precio de cada
#    pizza es casi fijo en el tiempo: la variación de precio es
#    *entre productos*, no dentro del mismo producto a lo largo del año)
# ---------------------------------------------------------------------------
def agregar_por_producto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa el detalle de órdenes a nivel de variante de pizza
    (nombre + tamaño), calculando precio promedio y cantidad total vendida.
    Esta tabla es la base para estimar la curva de demanda (precio vs. cantidad).
    """
    agg = (
        df.groupby(["pizza_name_id", "pizza_name", "pizza_category", "pizza_size"])
        .agg(
            precio_promedio=("unit_price", "mean"),
            cantidad_total=("quantity", "sum"),
            ingreso_total=("total_price", "sum"),
            n_ordenes=("order_id", "nunique"),
        )
        .reset_index()
    )
    agg["ln_precio"] = np.log(agg["precio_promedio"])
    agg["ln_cantidad"] = np.log(agg["cantidad_total"])
    return agg


# ---------------------------------------------------------------------------
# 3. Modelo econométrico: elasticidad-precio vía regresión log-log
#    ln(Q) = b0 + b1*ln(P) + controles categóricos   ->   b1 = elasticidad
# ---------------------------------------------------------------------------
def estimar_elasticidad_global(agg: pd.DataFrame):
    """Ajusta un modelo log-log controlando por categoría y tamaño."""
    modelo = smf.ols(
        "ln_cantidad ~ ln_precio + C(pizza_category) + C(pizza_size)",
        data=agg,
    ).fit()
    elasticidad = modelo.params["ln_precio"]
    return modelo, elasticidad


def estimar_elasticidad_por_categoria(agg: pd.DataFrame) -> pd.DataFrame:
    """Ajusta un log-log independiente para cada categoría de pizza."""
    resultados = []
    for categoria, sub in agg.groupby("pizza_category"):
        if len(sub) < 5:
            continue
        modelo = smf.ols("ln_cantidad ~ ln_precio", data=sub).fit()
        resultados.append(
            {
                "pizza_category": categoria,
                "elasticidad": modelo.params["ln_precio"],
                "p_valor": modelo.pvalues["ln_precio"],
                "r2": modelo.rsquared,
                "n_productos": len(sub),
            }
        )
    return pd.DataFrame(resultados).sort_values("elasticidad")


def interpretar_elasticidad(valor: float) -> str:
    """Traduce el coeficiente a una interpretación de negocio en español."""
    e = abs(valor)
    if e > 1:
        tipo = "elástica"
        detalle = "un aumento de precio reduce la cantidad demandada más que proporcionalmente"
    elif e < 1:
        tipo = "inelástica"
        detalle = "la cantidad demandada reacciona menos que proporcionalmente al precio"
    else:
        tipo = "unitaria"
        detalle = "la cantidad demandada cambia en la misma proporción que el precio"
    return f"Demanda {tipo} (|E|={e:.2f}): {detalle}."


# ---------------------------------------------------------------------------
# 4. Modelo predictivo (Random Forest) a nivel de línea de orden.
#    Complementa al modelo econométrico: predice cantidad esperada dada
#    una combinación de precio, categoría, tamaño y estacionalidad.
# ---------------------------------------------------------------------------
def entrenar_modelo_predictivo(df: pd.DataFrame):
    datos = df.copy()
    datos = datos.dropna(subset=["dia_semana", "mes"])

    X = pd.get_dummies(
        datos[["unit_price", "pizza_category", "pizza_size", "dia_semana", "mes"]],
        columns=["pizza_category", "pizza_size", "dia_semana"],
        drop_first=True,
    )
    y = datos["quantity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    modelo.fit(X_train, y_train)

    pred = modelo.predict(X_test)
    metricas = {
        "MAE": mean_absolute_error(y_test, pred),
        "R2": r2_score(y_test, pred),
    }
    return modelo, list(X.columns), metricas


# ---------------------------------------------------------------------------
# 5. Simulación de cambio de precio (insumo directo para la app Streamlit)
# ---------------------------------------------------------------------------
def simular_cambio_precio(precio_base: float, cantidad_base: float,
                           elasticidad: float, variacion_pct: float):
    """
    Aplica la definición de elasticidad-precio para simular el efecto
    de una variación porcentual de precio sobre la cantidad y el ingreso.

        %ΔQ = elasticidad * %ΔP
    """
    nuevo_precio = precio_base * (1 + variacion_pct)
    variacion_cantidad_pct = elasticidad * variacion_pct
    nueva_cantidad = cantidad_base * (1 + variacion_cantidad_pct)

    ingreso_base = precio_base * cantidad_base
    nuevo_ingreso = nuevo_precio * nueva_cantidad

    return {
        "precio_base": precio_base,
        "nuevo_precio": nuevo_precio,
        "cantidad_base": cantidad_base,
        "nueva_cantidad": nueva_cantidad,
        "variacion_cantidad_pct": variacion_cantidad_pct,
        "ingreso_base": ingreso_base,
        "nuevo_ingreso": nuevo_ingreso,
        "variacion_ingreso_pct": (nuevo_ingreso - ingreso_base) / ingreso_base,
    }
