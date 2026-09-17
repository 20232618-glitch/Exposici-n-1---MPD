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

tab_variables, tab_exploracion, tab_simulador, tab_optimizador, tab_opti1 = st.tabs(
    ["Análisis de Variables", "📊 Exploración de datos", "🎛️ Simulador de precio", "🎯 Optimizador de precio","🎯 Optimizador por historia"]
)

# ---------------------------------------------------------------------------
# TAB 0 — Análisis Descriptivo (Variables de la Dimensión Precio)
# ---------------------------------------------------------------------------
with tab_variables:
    st.subheader("Análisis descriptivo — Dimensión Precio y Demanda")
    st.caption(
        "Estadísticas e indicadores clave para el modelamiento de elasticidad: "
        "**precio unitario** (precio), **quantity** (demanda) y **total_price** (ingreso), "
        "junto con sus dimensiones de segmentación (**pizza_category** y **pizza_size**)."
    )

    # -----------------------------------------------------------------------
    # 1. Tabla de Estadísticos Descriptivos (Numéricas)
    # -----------------------------------------------------------------------
    st.markdown("#### 1. Estadísticos descriptivos — Variables numéricas")
    vars_num = ["unit_price", "quantity", "total_price"]
    
    # Cálculo exacto de media, desviación, min, max y moda
    desc_df = df[vars_num].agg(["mean", "std", "min", "max"]).T
    modas = df[vars_num].mode().iloc[0]
    desc_df.insert(0, "moda", modas)
    desc_df.columns = ["moda", "media", "desv_estandar", "minimo", "maximo"]
    
    st.dataframe(
        desc_df.style.format({
            "moda": "{:.2f}",
            "media": "{:.4f}",
            "desv_estandar": "{:.4f}",
            "minimo": "{:.2f}",
            "maximo": "{:.2f}"
        }),
        use_container_width=True
    )

    # -----------------------------------------------------------------------
    # 2. Resumen de Variables Categóricas
    # -----------------------------------------------------------------------
    st.markdown("#### 2. Distribución de variables categóricas de segmentación")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown(f"**pizza_category** (Moda: `{df['pizza_category'].mode()[0]}`)")
        st.dataframe(df["pizza_category"].value_counts().rename("Frecuencia"), use_container_width=True)
    with col_c2:
        st.markdown(f"**pizza_size** (Moda: `{df['pizza_size'].mode()[0]}`)")
        st.dataframe(df["pizza_size"].value_counts().rename("Frecuencia"), use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # 3. Histogramas de Comportamiento
    # -----------------------------------------------------------------------
    st.markdown("#### 3. Histogramas y densidad de las variables clave")
    sns.set_theme(style="whitegrid")
    
    fig_hist, ax_hist = plt.subplots(1, 3, figsize=(15, 4))
    
    sns.histplot(df["unit_price"], bins=20, kde=True, ax=ax_hist[0])
    ax_hist[0].set_title("Precio unitario")
    ax_hist[0].set_xlabel("unit_price")
    ax_hist[0].set_ylabel("Count")

    sns.histplot(df["quantity"], bins=10, ax=ax_hist[1])
    ax_hist[1].set_title("Cantidad por línea de orden")
    ax_hist[1].set_xlabel("quantity")
    ax_hist[1].set_ylabel("Count")

    sns.histplot(df["total_price"], bins=20, kde=True, ax=ax_hist[2])
    ax_hist[2].set_title("Precio total")
    ax_hist[2].set_xlabel("total_price")
    ax_hist[2].set_ylabel("Count")

    fig_hist.tight_layout()
    st.pyplot(fig_hist)
    plt.close(fig_hist)

    # -----------------------------------------------------------------------
    # 4. Boxplots de Dispersión
    # -----------------------------------------------------------------------
    st.markdown("#### 4. Dispersión y detección de valores atípicos")
    fig_box, ax_box = plt.subplots(1, 2, figsize=(12, 4))
    
    sns.boxplot(y=df["unit_price"], ax=ax_box[0])
    ax_box[0].set_title("Dispersión del precio unitario")
    ax_box[0].set_ylabel("unit_price")

    sns.boxplot(y=df["total_price"], ax=ax_box[1])
    ax_box[1].set_title("Dispersión del precio total")
    ax_box[1].set_ylabel("total_price")

    fig_box.tight_layout()
    st.pyplot(fig_box)
    plt.close(fig_box)

    # -----------------------------------------------------------------------
    # 5. Precio Promedio por Categoría
    # -----------------------------------------------------------------------
    st.markdown("#### 5. Precio unitario promedio por categoría")
    col_bar1, col_bar2 = st.columns([1.2, 1])
    
    with col_bar1:
        fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
        sns.barplot(
            data=df, 
            x="pizza_category", 
            y="unit_price", 
            estimator="mean", 
            errorbar="sd", 
            color="#2b7bba", 
            ax=ax_bar
        )
        ax_bar.set_title("Precio unitario promedio por categoría (con desviación estándar)")
        ax_bar.set_xlabel("pizza_category")
        ax_bar.set_ylabel("unit_price")
        ax_bar.tick_params(axis="x", rotation=20)
        fig_bar.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    with col_bar2:
        st.markdown("**Hallazgos de la distribución:**")
        st.write(
            "- **Asimetría en cantidad:** La gran mayoría de pedidos se registran por exactamente 1 unidad "
            f"(media = {df['quantity'].mean():.2f}), concentrando los picos de demanda individual."
        )
        st.write(
            "- **Outliers en ingreso:** La dispersión de `total_price` presenta valores atípicos que superan "
            f"los $60 hasta el máximo de ${df['total_price'].max():.2f}, producto de compras con múltiples cantidades."
        )
        st.write(
            "- **Segmentación de precios:** La categoría `Classic` mantiene el ticket promedio más bajo, "
            "mientras que `Chicken` y `Supreme` lideran los precios unitarios promedio."
        )

# ---------------------------------------------------------------------------
# TAB 1 — Exploración (nivel 1 de prototipo: informativo)
# ---------------------------------------------------------------------------
with tab_exploracion:
    col1, col2, col3 = st.columns(3)
    col1.metric("Variantes de pizza", f"{len(agg):,}")
    col2.metric("Órdenes totales", f"{df['order_id'].nunique():,}")
    col3.metric("Ingreso total", f"${df['total_price'].sum():,.0f}")

    st.subheader("Curva de demanda desagregada (Control por tamaño)")
    st.caption("Aísla el efecto del volumen de pizzas pequeñas (S), medianas (M) y grandes (L) para evitar sesgos de escala.")

    agg_filtrado = agg[agg["pizza_size"].isin(["S", "M", "L"])].copy()
    
    g_tam = sns.lmplot(
        data=agg_filtrado,
        x="ln_precio",
        y="ln_cantidad",
        hue="pizza_category",
        col="pizza_size",
        col_order=["S", "M", "L"],
        height=3.8,
        aspect=1.05,
        scatter_kws={"alpha": 0.6, "s": 35},
        sharey=False,
        sharex=False
    )
    g_tam.fig.subplots_adjust(top=0.85)
    g_tam.fig.suptitle("Relación log-log controlando por tamaño (S, M, L)", fontsize=13, weight="bold")
    g_tam.set_axis_labels("ln(precio promedio)", "ln(cantidad total vendida)")
    st.pyplot(g_tam.figure)
    plt.close("all")
    
# -----------------------------------------------------------------------
# 1.1 GRÁFICO GENERAL (Todas las categorías y tamaños juntos)
# -----------------------------------------------------------------------
    st.subheader("Curva de demanda global (log-precio vs. log-cantidad por categoría)")
    st.caption("Muestra la distribución total agregada de precios y cantidades vendidas por cada categoría.")
    
    sns.set_theme(style="whitegrid")
    g_gen = sns.lmplot(
        data=agg,
        x="ln_precio",
        y="ln_cantidad",
        hue="pizza_category",
        height=4.8,
        aspect=1.4,
        scatter_kws={"alpha": 0.5, "s": 40}
    )
    g_gen.fig.subplots_adjust(top=0.92)
    g_gen.fig.suptitle("Modelo de regresión log-log: precio vs. demanda por categoría (Global)", fontsize=13, weight="bold")
    g_gen.set_axis_labels("ln(precio promedio)", "ln(cantidad total vendida)")
    st.pyplot(g_gen.figure)
    plt.close("all")
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
    st.subheader("🎛️ Simulador interactivo de sensibilidad de demanda")
    st.caption(
        "Simula el impacto de variaciones en el precio unitario sobre el volumen de ventas "
        "y la recaudación bruta esperada."
    )

    # 1. Validación de la elasticidad global controlada
    st.markdown(
        f"**Elasticidad-precio global del modelo (controlando por categoría y tamaño): `{elasticidad_global:.3f}`**  \n"
        f"{interpretar_elasticidad(elasticidad_global)}"
    )

    col_izq, col_der = st.columns([1, 2])

    with col_izq:
        pizza_sel = st.selectbox(
            "Selecciona una variante de pizza",
            options=agg["pizza_name"] + " (" + agg["pizza_size"] + ")",
            key="sim_pizza_sel"
        )
        
        # Dejamos por defecto False para usar el modelo global consistente
        usar_elasticidad_categoria = st.checkbox(
            "Usar elasticidad específica de la categoría",
            value=False,
            help="Atención: la elasticidad por categoría sin desagregar tamaño puede presentar sesgos estadísticos."
        )
        
        variacion_pct = st.slider(
            "Variación de precio a simular (%)", 
            min_value=-30, 
            max_value=30,
            value=10, 
            step=1,
            key="sim_slider_pct"
        ) / 100

    # Fila de datos del producto base seleccionado
    fila = agg[(agg["pizza_name"] + " (" + agg["pizza_size"] + ")") == pizza_sel].iloc[0]

    # Determinación y saneamiento de la elasticidad a aplicar
    elasticidad_usar = elasticidad_global
    advertencia_consistencia = None

    if usar_elasticidad_categoria:
        coincidencia = tabla_categorias[tabla_categorias["pizza_category"] == fila["pizza_category"]]
        if not coincidencia.empty:
            e_cat = coincidencia.iloc[0]["elasticidad"]
            # Salvaguarda: la ley de la demanda exige pendiente negativa
            if e_cat >= 0:
                advertencia_consistencia = (
                    f"La elasticidad no controlada de la categoría **{fila['pizza_category']}** es positiva ({e_cat:.3f}) "
                    f"debido al sesgo de volumen por tamaño de pizza. Para mantener coherencia económica con la ley de la demanda, "
                    f"el simulador aplica la elasticidad global controlada ({elasticidad_global:.3f})."
                )
                elasticidad_usar = elasticidad_global
            else:
                elasticidad_usar = e_cat

    # Cálculo paramétrico del simulador
    resultado = simular_cambio_precio(
        precio_base=fila["precio_promedio"],
        cantidad_base=fila["cantidad_total"],
        elasticidad=elasticidad_usar,
        variacion_pct=variacion_pct,
    )

    with col_der:
        st.markdown(f"**Elasticidad aplicada: `{elasticidad_usar:.3f}`**")
        
        if advertencia_consistencia:
            st.warning(advertencia_consistencia)

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Nuevo Precio", 
            f"${resultado['nuevo_precio']:.2f}",
            delta=f"{variacion_pct*100:+.0f}%"
        )
        m2.metric(
            "Demanda Estimada (Q)", 
            f"{resultado['nueva_cantidad']:.0f} u.",
            delta=f"{resultado['variacion_cantidad_pct']*100:+.1f}%"
        )
        m3.metric(
            "Ingreso Estimado (P × Q)", 
            f"${resultado['nuevo_ingreso']:,.0f}",
            delta=f"{resultado['variacion_ingreso_pct']*100:+.1f}%"
        )

        # Diagnóstico analítico de negocio
        st.markdown("---")
        if variacion_pct > 0:
            if resultado['variacion_cantidad_pct'] < 0 and resultado['variacion_ingreso_pct'] > 0:
                st.success(
                    f"✔ **Comportamiento coherente (Demanda Inelástica):** Al subir el precio un **{variacion_pct*100:+.0f}%**, "
                    f"la cantidad demandada cae un **{abs(resultado['variacion_cantidad_pct']*100):.1f}%** (respetando la ley de la demanda). "
                    f"Sin embargo, el ingreso total se incrementa en **{resultado['variacion_ingreso_pct']*100:+.1f}%** porque el mayor precio unitario "
                    f"absorbe la ligera contracción de pedidos."
                )
            elif resultado['variacion_cantidad_pct'] < 0 and resultado['variacion_ingreso_pct'] <= 0:
                st.info(
                    f"⚠ **Demanda Elástica:** La subida de precio provoca una fuga de clientes de **{abs(resultado['variacion_cantidad_pct']*100):.1f}%**, "
                    f"reduciendo la facturación total en **{resultado['variacion_ingreso_pct']*100:.1f}%**."
                )
        elif variacion_pct < 0:
            if resultado['variacion_cantidad_pct'] > 0 and resultado['variacion_ingreso_pct'] <= 0:
                st.info(
                    f"✔ **Comportamiento coherente:** Al bajar el precio un **{abs(variacion_pct*100):.0f}%**, "
                    f"las unidades vendidas aumentan un **{resultado['variacion_cantidad_pct']*100:+.1f}%**, pero no compensa el descuento, "
                    f"reduciendo el ingreso bruto en **{resultado['variacion_ingreso_pct']*100:.1f}%**."
                )
            else:
                st.success(
                    f"✔ El descuento en precio estimula el volumen de ventas en **{resultado['variacion_cantidad_pct']*100:+.1f}%**, aumentando la facturación."
                )

    st.caption(
        "Nota metodológica: El simulador asume elasticidad constante en el rango local analizado "
        "y se basa en la sensibilidad estimada controlando por tamaño y categoría."
    )
    
# ---------------------------------------------------------------------------
# TAB 3 — Optimizador basado en el Modelo y Simulador
# ---------------------------------------------------------------------------
with tab_optimizador:
  st.subheader("🎯 Optimización paramétrica de precio por variante")
  st.caption(
      "Encuentra el precio óptimo teórico proyectando la curva de ingresos"
      " mediante la elasticidad-precio del modelo."
  )

  # Selección por variante específica (alineado al simulador)
  opciones_variantes = agg["pizza_name"] + " (" + agg["pizza_size"] + ")"
  pizza_sel_opt = st.selectbox(
      "Selecciona la variante de pizza a optimizar:",
      options=opciones_variantes,
      key="opt_var_tab3",
  )

  fila_opt = agg[opciones_variantes == pizza_sel_opt].iloc[0]
  p0 = fila_opt["precio_promedio"]
  q0 = fila_opt["cantidad_total"]
  cat_opt = fila_opt["pizza_category"]

  # Selección y saneamiento de elasticidad para garantizar coherencia económica
  coincidencia = tabla_categorias[tabla_categorias["pizza_category"] == cat_opt]
  e_cat = (
      coincidencia.iloc[0]["elasticidad"]
      if not coincidencia.empty
      else elasticidad_global
  )

  # Si la elasticidad de la categoría es positiva (anomalía muestral), usamos la global controlada
  if e_cat >= 0:
    elasticidad_usar_opt = elasticidad_global
    aviso_elasticidad = (
        f"La categoría **{cat_opt}** presenta una elasticidad muestral"
        f" positiva ({e_cat:.3f}). Se aplica la elasticidad global controlada"
        f" ({elasticidad_global:.3f}) para respetar la ley de la demanda."
    )
  else:
    elasticidad_usar_opt = e_cat
    aviso_elasticidad = None

  # Barrido continuo de precios (-40% a +40%) usando simular_cambio_precio
  variaciones = [v / 100 for v in range(-40, 41, 1)]
  sims = []
  for var in variaciones:
    res = simular_cambio_precio(
        precio_base=p0,
        cantidad_base=q0,
        elasticidad=elasticidad_usar_opt,
        variacion_pct=var,
    )
    sims.append({
        "variacion_pct": var * 100,
        "precio": res["nuevo_precio"],
        "cantidad": res["nueva_cantidad"],
        "ingreso": res["nuevo_ingreso"],
    })

  df_curva = pd.DataFrame(sims)
  punto_opt = df_curva.loc[df_curva["ingreso"].idxmax()]

  if aviso_elasticidad:
    st.info(aviso_elasticidad)

  m1, m2, m3, m4 = st.columns(4)
  m1.metric("Precio Actual", f"${p0:.2f}")
  m2.metric(
      "Precio Óptimo Teórico",
      f"${punto_opt['precio']:.2f}",
      delta=f"{punto_opt['variacion_pct']:+.0f}% sugerido",
  )
  m3.metric(
      "Ingreso Máx. Proyectado",
      f"${punto_opt['ingreso']:,.0f}",
      delta=(
          f"{((punto_opt['ingreso'] / (p0 * q0)) - 1)*100:+.1f}% vs. actual"
      ),
  )
  m4.metric("Elasticidad (ε)", f"{elasticidad_usar_opt:.3f}")

  # Gráfico estilizado y limpio de la curva de ingresos
  fig_curva, ax_curva = plt.subplots(figsize=(8, 3.8))
  sns.lineplot(
      data=df_curva,
      x="precio",
      y="ingreso",
      color="#1f77b4",
      linewidth=2.5,
      ax=ax_curva,
  )
  ax_curva.axvline(
      punto_opt["precio"],
      color="red",
      linestyle="--",
      linewidth=1.8,
      label=f"Óptimo sugerido: ${punto_opt['precio']:.2f}",
  )
  ax_curva.axvline(
      p0,
      color="gray",
      linestyle=":",
      linewidth=1.5,
      label=f"Precio actual: ${p0:.2f}",
  )
  ax_curva.set_title(
      f"Curva de Ingreso Proyectado vs. Precio — {pizza_sel_opt}",
      fontsize=11,
      weight="bold",
  )
  ax_curva.set_xlabel("Precio Simulado ($)")
  ax_curva.set_ylabel("Ingreso Estimado ($)")
  ax_curva.legend(loc="best")
  st.pyplot(fig_curva)
  plt.close(fig_curva)

  # Diagnóstico interpretativo
  if elasticidad_usar_opt > -1:
    st.success(
        f"📌 **Demanda Inelástica ({elasticidad_usar_opt:.3f}):** El cliente tolera"
        " incrementos sin que el volumen caiga drásticamente. El modelo ubica"
        f" el óptimo en **${punto_opt['precio']:.2f}** dentro del rango"
        " comercial seguro."
    )
  else:
    st.warning(
        f"📌 **Demanda Elástica ({elasticidad_usar_opt:.3f}):** Clientes sensibles"
        f" al precio. Superar los **${punto_opt['precio']:.2f}** destruirá"
        " demanda e ingresos brutos."
    )
            
# ---------------------------------------------------------------------------
# TAB 4 — Optimizador por Historia y Desempeño Real de la Variante
# ---------------------------------------------------------------------------
with tab_opti1:
  st.subheader("🏆 Desempeño histórico y precio de referencia por variante")
  st.caption(
      "Audita el rendimiento real de la variante seleccionada frente a sus"
      " pares directos de la misma categoría y tamaño."
  )

  opciones_variantes_h = agg["pizza_name"] + " (" + agg["pizza_size"] + ")"
  pizza_sel_h = st.selectbox(
      "Selecciona la variante a auditar:",
      options=opciones_variantes_h,
      key="opt_var_tab4",
  )

  fila_h = agg[opciones_variantes_h == pizza_sel_h].iloc[0]
  cat_h = fila_h["pizza_category"]
  tam_h = fila_h["pizza_size"]

  # Filtrar el grupo competitivo directo (mismo tamaño y misma categoría)
  grupo_pares = agg[
      (agg["pizza_category"] == cat_h) & (agg["pizza_size"] == tam_h)
  ].copy()
  grupo_pares["ingreso_total"] = (
      grupo_pares["precio_promedio"] * grupo_pares["cantidad_total"]
  )
  grupo_pares = grupo_pares.sort_values(
      "ingreso_total", ascending=False
  ).reset_index(drop=True)

  lider_grupo = grupo_pares.iloc[0]
  ingreso_actual_var = fila_h["precio_promedio"] * fila_h["cantidad_total"]

  col_h1, col_h2, col_h3 = st.columns(3)
  col_h1.metric("Precio de esta Variante", f"${fila_h['precio_promedio']:.2f}")
  col_h2.metric("Ingreso Histórico Generado", f"${ingreso_actual_var:,.0f}")
  col_h3.metric(
      "Precio del Líder del Segmento",
      f"${lider_grupo['precio_promedio']:.2f}",
      delta=f"Líder: {lider_grupo['pizza_name']}",
  )

  st.markdown("---")
  st.markdown(
      f"**Segmento:** Categoría `{cat_h}` | Tamaño `{tam_h}` (Evaluado entre"
      f" {len(grupo_pares)} variantes competidoras directas)"
  )

  # Gráfico comparativo usando directamente ax.scatter para evitar conflictos de leyenda
  fig_comp, ax_comp = plt.subplots(figsize=(8, 4))
  grupo_pares["es_seleccionada"] = (
      grupo_pares["pizza_name"] == fila_h["pizza_name"]
  )

  df_otros = grupo_pares[~grupo_pares["es_seleccionada"]]
  df_sel = grupo_pares[grupo_pares["es_seleccionada"]]

  # Puntos de competidores
  if not df_otros.empty:
    ax_comp.scatter(
        df_otros["precio_promedio"],
        df_otros["ingreso_total"],
        s=df_otros["cantidad_total"] / 4,
        color="#1976d2",
        alpha=0.6,
        edgecolors="none",
        label="Otras variantes del segmento",
    )

  # Punto destacado de la variante seleccionada
  if not df_sel.empty:
    ax_comp.scatter(
        df_sel["precio_promedio"],
        df_sel["ingreso_total"],
        s=df_sel["cantidad_total"] / 3 + 100,
        color="#d32f2f",
        marker="*",
        edgecolors="black",
        linewidth=0.8,
        label=f"Seleccionada: {fila_h['pizza_name']}",
        zorder=5,
    )

  # Línea de referencia del líder del segmento
  ax_comp.axvline(
      lider_grupo["precio_promedio"],
      color="#2e7d32",
      linestyle="--",
      alpha=0.8,
      linewidth=1.8,
      label=f"Precio Líder: ${lider_grupo['precio_promedio']:.2f}",
  )

  ax_comp.set_title(
      f"Posición Competitiva en {cat_h} ({tam_h})", fontsize=11, weight="bold"
  )
  ax_comp.set_xlabel("Precio Promedio ($)")
  ax_comp.set_ylabel("Ingreso Total ($)")
  ax_comp.legend(loc="best")
  ax_comp.grid(True, linestyle=":", alpha=0.6)

  st.pyplot(fig_comp)
  plt.close(fig_comp)

  # Tabla del segmento
  st.write("Variantes competidoras en este segmento:")
  st.dataframe(
      grupo_pares[[
          "pizza_name",
          "precio_promedio",
          "cantidad_total",
          "ingreso_total",
      ]],
      width="stretch",
  )
