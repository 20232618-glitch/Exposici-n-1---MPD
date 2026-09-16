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

tab_exploracion, tab_simulador, tab_optimizador, tab_opti1 = st.tabs(
    ["📊 Exploración de datos", "🎛️ Simulador de precio", "🎯 Optimizador de precio","Optimizador por historia"]
)
# ---------------------------------------------------------------------------
# TAB 1 — Exploración (nivel 1 de prototipo: informativo)
# ---------------------------------------------------------------------------
with tab_exploracion:
    col1, col2, col3 = st.columns(3)
    col1.metric("Variantes de pizza", f"{len(agg):,}")
    col2.metric("Órdenes totales", f"{df['order_id'].nunique():,}")
    col3.metric("Ingreso total", f"${df['total_price'].sum():,.0f}")

    st.subheader("Curva de demanda (log-precio vs. log-cantidad por categoría)")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.set_theme(style="whitegrid")
    g = sns.lmplot(
        data=agg,
        x="ln_precio",
        y="ln_cantidad",
        hue="pizza_category",
        height=5,
        aspect=1.3,
        scatter_kws={"alpha": 0.5}
    )
    g.fig.subplots_adjust(top=0.92)
    g.fig.suptitle("Modelo de regresión log-log: precio vs. demanda por categoría")
    g.set_axis_labels("ln(precio promedio)", "ln(cantidad total vendida)")
    st.pyplot(g.figure)
    
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
    
# ---------------------------------------------------------------------------
# TAB 3 — Optimizador basado en el Modelo y Simulador
# ---------------------------------------------------------------------------
with tab_optimizador:
    st.subheader("🎯 Optimización paramétrica de precio (Modelo Econométrico)")
    st.caption(
        "Utiliza la elasticidad estimada por el modelo log-log para proyectar "
        "la curva de ingresos y encontrar el precio que maximiza las ventas."
    )

    col_cat, col_tam = st.columns(2)
    with col_cat:
        cat_sel = st.selectbox("1. Selecciona Categoría:", options=sorted(agg["pizza_category"].unique()), key="opt_cat")
    with col_tam:
        tamanos_disp = sorted(agg[agg["pizza_category"] == cat_sel]["pizza_size"].unique())
        tam_sel = st.selectbox("2. Selecciona Tamaño:", options=tamanos_disp, key="opt_tam")

    sub_df = agg[(agg["pizza_category"] == cat_sel) & (agg["pizza_size"] == tam_sel)].copy()

    if sub_df.empty:
        st.warning("No hay registros para este segmento.")
    else:
        # 1. Parámetros base del segmento (promedio ponderado o medio)
        precio_base_segmento = sub_df["precio_promedio"].mean()
        cantidad_base_segmento = sub_df["cantidad_total"].sum()

        # 2. Extraer elasticidad del modelo para esta categoría
        coincidencia = tabla_categorias[tabla_categorias["pizza_category"] == cat_sel]
        elasticidad_usar = coincidencia.iloc[0]["elasticidad"] if not coincidencia.empty else elasticidad_global

        # 3. Barrido de precios usando la función simular_cambio_precio (-50% a +50%)
        variaciones = [v / 100 for v in range(-50, 51, 1)]
        simulaciones = []

        for var in variaciones:
            res = simular_cambio_precio(
                precio_base=precio_base_segmento,
                cantidad_base=cantidad_base_segmento,
                elasticidad=elasticidad_usar,
                variacion_pct=var
            )
            simulaciones.append({
                "variacion_pct": var * 100,
                "precio": res["nuevo_precio"],
                "cantidad": res["nueva_cantidad"],
                "ingreso": res["nuevo_ingreso"]
            })

        df_curva = pd.DataFrame(simulaciones)

        # 4. Encontrar el punto de ingreso máximo según la simulación
        punto_optimo = df_curva.loc[df_curva["ingreso"].idxmax()]

        # Métricas resultantes
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precio Base Actual", f"${precio_base_segmento:.2f}")
        m2.metric(
            "Precio Óptimo Estimado", 
            f"${punto_optimo['precio']:.2f}",
            delta=f"{punto_optimo['variacion_pct']:+.0f}% sugerido"
        )
        m3.metric("Ingreso Proyectado Máx.", f"${punto_optimo['ingreso']:,.0f}")
        m4.metric("Elasticidad (ε)", f"{elasticidad_usar:.3f}")

        # 5. Gráfico de la curva teórica de ingresos
        fig_curva, ax_curva = plt.subplots(figsize=(8, 3.8))
        sns.lineplot(data=df_curva, x="precio", y="ingreso", color="#1f77b4", linewidth=2.5, ax=ax_curva)
        ax_curva.axvline(
            punto_optimo["precio"], 
            color="red", 
            linestyle="--", 
            label=f"Óptimo: ${punto_optimo['precio']:.2f}"
        )
        ax_curva.axvline(
            precio_base_segmento, 
            color="gray", 
            linestyle=":", 
            label=f"Actual: ${precio_base_segmento:.2f}"
        )
        ax_curva.set_title(f"Curva de Ingreso Proyectado vs. Precio ({cat_sel} - {tam_sel})")
        ax_curva.set_xlabel("Precio Simulado ($)")
        ax_curva.set_ylabel("Ingreso Proyectado ($)")
        ax_curva.legend()
        st.pyplot(fig_curva)

        # Diagnóstico analítico
        if elasticidad_usar > -1:
            st.info(
                f"📌 **Demanda Inelástica ({elasticidad_usar:.2f}):** Según el modelo, el mercado tolera "
                f"incrementos de precio sin perder ingresos. El simulador ubica la optimización en "
                f"**${punto_optimo['precio']:.2f}** dentro del rango testeado."
            )
        else:
            st.warning(
                f"📌 **Demanda Elástica ({elasticidad_usar:.2f}):** El consumidor es sensible a aumentos. "
                f"Subir el precio por encima de **${punto_optimo['precio']:.2f}** destruirá demanda e ingreso total."
            )
            
# ---------------------------------------------------------------------------
# TAB 4 — Optimizador de precio por categoría y tamaño
# ---------------------------------------------------------------------------
with tab_opti1:
    st.subheader("Búsqueda de precio óptimo por segmento")
    st.caption("Filtra por categoría y tamaño para encontrar el rango de precio que maximiza los ingresos.")

    col_cat, col_tam = st.columns(2)
    
    with col_cat:
        cat_sel = st.selectbox("Categoría:", options=sorted(agg["pizza_category"].unique()))
    with col_tam:
        # Filtra los tamaños disponibles para esa categoría
        tamanos_disp = sorted(agg[agg["pizza_category"] == cat_sel]["pizza_size"].unique())
        tam_sel = st.selectbox("Tamaño:", options=tamanos_disp)

    # Filtrar el dataframe según la selección
    sub_df = agg[(agg["pizza_category"] == cat_sel) & (agg["pizza_size"] == tam_sel)].copy()

    if sub_df.empty:
        st.warning("No hay datos disponibles para la combinación seleccionada.")
    else:
        # Calcular ingreso por variante observada
        sub_df["ingreso_total"] = sub_df["precio_promedio"] * sub_df["cantidad_total"]
        
        # Variante con mejor rendimiento histórico
        mejor_variante = sub_df.sort_values("ingreso_total", ascending=False).iloc[0]
        precio_optimo_historico = mejor_variante["precio_promedio"]
        
        # Recuperar elasticidad de la categoría seleccionada
        coincidencia = tabla_categorias[tabla_categorias["pizza_category"] == cat_sel]
        elasticidad_cat = coincidencia.iloc[0]["elasticidad"] if not coincidencia.empty else elasticidad_global

        # Métricas principales
        m1, m2, m3 = st.columns(3)
        m1.metric("Precio óptimo observado", f"${precio_optimo_historico:.2f}")
        m2.metric("Ingreso máx. histórico", f"${mejor_variante['ingreso_total']:,.0f}")
        m3.metric("Elasticidad de la categoría", f"{elasticidad_cat:.2f}")

        # Recomendación comercial basada en elasticidad
        st.markdown("---")
        st.markdown("**Diagnóstico del segmento:**")
        if elasticidad_cat < -1:
            st.warning(
                f"La categoría **{cat_sel}** tiene demanda elástica ({elasticidad_cat:.2f}). "
                f"El mercado es sensible al precio: subirlo más allá de **${precio_optimo_historico:.2f}** "
                f"provocará caídas pronunciadas en la demanda que reducirán el ingreso total."
            )
        elif -1 <= elasticidad_cat <= 0:
            st.success(
                f"La categoría **{cat_sel}** tiene demanda inelástica ({elasticidad_cat:.2f}). "
                f"Los clientes toleran precios mayores: existe margen para testear incrementos por encima de "
                f"**${precio_optimo_historico:.2f}** sin sacrificar ingresos brutos."
            )
        else:
            st.info(
                f"Elasticidad observada: {elasticidad_cat:.2f}. El precio óptimo de referencia para "
                f"este segmento es **${precio_optimo_historico:.2f}** (logrado por la variante `{mejor_variante['pizza_name']}`)."
            )

        # Gráfico comparativo de variantes dentro del grupo
        fig_opt, ax_opt = plt.subplots(figsize=(8, 4))
        sns.scatterplot(
            data=sub_df,
            x="precio_promedio",
            y="ingreso_total",
            size="cantidad_total",
            sizes=(50, 400),
            hue="pizza_name",
            legend=False,
            ax=ax_opt
        )
        ax_opt.axvline(precio_optimo_historico, color="red", linestyle="--", alpha=0.7, label=f"Óptimo: ${precio_optimo_historico:.2f}")
        ax_opt.set_title(f"Ingresos vs. Precio — {cat_sel} ({tam_sel})")
        ax_opt.set_xlabel("Precio promedio ($)")
        ax_opt.set_ylabel("Ingreso total ($)")
        ax_opt.legend()
        st.pyplot(fig_opt)

        # Tabla de variantes en ese segmento
        st.write("Variantes evaluadas en este segmento:")
        st.dataframe(
            sub_df[["pizza_name", "precio_promedio", "cantidad_total", "ingreso_total"]]
            .sort_values("ingreso_total", ascending=False),
            use_container_width=True
        )
