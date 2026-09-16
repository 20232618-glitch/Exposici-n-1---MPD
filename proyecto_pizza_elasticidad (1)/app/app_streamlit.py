# -*- coding: utf-8 -*-
"""
app_streamlit.py
Prototipo interactivo: Análisis de elasticidad-precio — Pizza Sales Dataset

Para ejecutar (desde Anaconda Prompt, ubicado en la carpeta /app):
    streamlit run app_streamlit.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils_modelo import (
    cargar_datos, agregar_por_producto, estimar_elasticidad_global,
    estimar_elasticidad_por_categoria, interpretar_elasticidad,
    simular_cambio_precio,
)

st.set_page_config(page_title="Elasticidad-precio | Pizza Sales", layout="wide")

# ---------------------------------------------------------------------------
# Carga de datos (cacheada para no recalcular en cada interacción)
# ---------------------------------------------------------------------------
RUTA_CSV_DEFECTO = os.path.join(os.path.dirname(__file__), "..", "data", "pizza_sales.csv")


@st.cache_data
def cargar_y_preparar(ruta_csv):
    df = cargar_datos(ruta_csv)
    agg = agregar_por_producto(df)
    return df, agg


@st.cache_resource
def ajustar_modelos(agg):
    modelo_global, elasticidad_global = estimar_elasticidad_global(agg)
    tabla_categorias = estimar_elasticidad_por_categoria(agg)
    return modelo_global, elasticidad_global, tabla_categorias


st.title("🍕 Análisis de elasticidad-precio y optimización de ventas")
st.caption(
    "Curso: Modelamiento Predictivo de Datos — Dataset: Pizza Sales Dataset (Kaggle)"
)

archivo_subido = st.sidebar.file_uploader("Cargar pizza_sales.csv", type="csv")

try:
    if archivo_subido is not None:
        df, agg = cargar_y_preparar(archivo_subido)
    else:
        df, agg = cargar_y_preparar(RUTA_CSV_DEFECTO)
except FileNotFoundError:
    st.warning(
        "No se encontró `data/pizza_sales.csv`. Descarga el dataset desde Kaggle "
        "(nextmillionaire/pizza-sales-dataset) y colócalo en esa ruta, o cárgalo "
        "manualmente con el botón de la izquierda."
    )
    st.stop()

modelo_global, elasticidad_global, tabla_categorias = ajustar_modelos(agg)

tab_exploracion, tab_simulador = st.tabs(["📊 Exploración de datos", "🎛️ Simulador de precio"])

# ---------------------------------------------------------------------------
# TAB 1 — Exploración (nivel 1 de prototipo: informativo)
# ---------------------------------------------------------------------------
with tab_exploracion:
    col1, col2, col3 = st.columns(3)
    col1.metric("Variantes de pizza", f"{len(agg):,}")
    col2.metric("Órdenes totales", f"{df['order_id'].nunique():,}")
    col3.metric("Ingreso total", f"${df['total_price'].sum():,.0f}")

    st.subheader("Curva de demanda (log-precio vs. log-cantidad)")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.regplot(data=agg, x="ln_precio", y="ln_cantidad", hue="pizza_category", ci=95,
                scatter_kws={"alpha": 0.5}, ax=ax)
    ax.set_xlabel("ln(precio promedio)")
    ax.set_ylabel("ln(cantidad total vendida)")
    st.pyplot(fig)

    st.subheader("Elasticidad-precio estimada por categoría")
    st.dataframe(tabla_categorias, use_container_width=True)

    st.subheader("Top 10 variantes por cantidad vendida")
    st.dataframe(
        agg.sort_values("cantidad_total", ascending=False)
        .head(10)[["pizza_name", "pizza_category", "pizza_size", "precio_promedio", "cantidad_total"]],
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# TAB 2 — Simulador interactivo (nivel 2 de prototipo: widgets controlados)
# ---------------------------------------------------------------------------
with tab_simulador:
    st.markdown(
        f"**Elasticidad-precio global del modelo (log-log, controlando por "
        f"categoría y tamaño): `{elasticidad_global:.3f}`**  \n"
        f"{interpretar_elasticidad(elasticidad_global)}"
    )

    col_izq, col_der = st.columns([1, 2])

    with col_izq:
        pizza_sel = st.selectbox(
            "Selecciona una variante de pizza",
            options=agg["pizza_name"] + " (" + agg["pizza_size"] + ")",
        )
        usar_elasticidad_categoria = st.checkbox(
            "Usar elasticidad específica de la categoría (en vez de la global)",
            value=True,
        )
        variacion_pct = st.slider(
            "Variación de precio a simular (%)", min_value=-30, max_value=30,
            value=10, step=1,
        ) / 100

    fila = agg[(agg["pizza_name"] + " (" + agg["pizza_size"] + ")") == pizza_sel].iloc[0]

    elasticidad_usar = elasticidad_global
    if usar_elasticidad_categoria:
        coincidencia = tabla_categorias[tabla_categorias["pizza_category"] == fila["pizza_category"]]
        if not coincidencia.empty:
            elasticidad_usar = coincidencia.iloc[0]["elasticidad"]

    resultado = simular_cambio_precio(
        precio_base=fila["precio_promedio"],
        cantidad_base=fila["cantidad_total"],
        elasticidad=elasticidad_usar,
        variacion_pct=variacion_pct,
    )

    with col_der:
        st.markdown(f"**Elasticidad aplicada a esta simulación: `{elasticidad_usar:.3f}`**")
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Precio", f"${resultado['nuevo_precio']:.2f}",
            delta=f"{variacion_pct*100:+.0f}%",
        )
        m2.metric(
            "Cantidad estimada", f"{resultado['nueva_cantidad']:.0f}",
            delta=f"{resultado['variacion_cantidad_pct']*100:+.1f}%",
        )
        m3.metric(
            "Ingreso estimado", f"${resultado['nuevo_ingreso']:,.0f}",
            delta=f"{resultado['variacion_ingreso_pct']*100:+.1f}%",
        )

        if resultado["variacion_ingreso_pct"] > 0:
            st.success(
                "Bajo el supuesto de elasticidad constante, este cambio de precio "
                "incrementaría el ingreso esperado de esta variante."
            )
        else:
            st.info(
                "Bajo el supuesto de elasticidad constante, este cambio de precio "
                "reduciría el ingreso esperado de esta variante."
            )

    st.caption(
        "Nota metodológica: la simulación asume elasticidad constante en el rango "
        "analizado y se basa en variación de precio *entre* variantes de pizza, no en "
        "un experimento temporal de precios sobre el mismo producto."
    )
