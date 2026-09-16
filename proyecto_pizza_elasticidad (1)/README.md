# Análisis de elasticidad-precio — Pizza Sales Dataset

Prototipo para el curso de Modelamiento Predictivo de Datos, siguiendo el flujo
paper → notebook → script `.py` → Streamlit.

## Estructura del proyecto
```
proyecto_pizza/
├── data/
│   └── pizza_sales.csv        <- colocar aquí el CSV descargado de Kaggle
├── notebooks/
│   └── 01_modelo_elasticidad_precio.ipynb
├── app/
│   └── app_streamlit.py
├── utils_modelo.py            <- funciones compartidas (modelo + simulación)
├── requirements.txt
└── README.md
```

## 1. Descargar el dataset
Descarga el CSV desde Kaggle y guárdalo como `data/pizza_sales.csv`:
https://www.kaggle.com/datasets/nextmillionaire/pizza-sales-dataset

## 2. Crear el entorno en Anaconda Prompt
Abre **Anaconda Prompt** y ubícate en la carpeta del proyecto:
```bat
cd ruta\a\proyecto_pizza
conda create -n pizza_elasticidad python=3.11 -y
conda activate pizza_elasticidad
pip install -r requirements.txt
```

## 3. Ejecutar el notebook exploratorio
```bat
jupyter notebook notebooks/01_modelo_elasticidad_precio.ipynb
```
Ejecuta las celdas en orden. Ahí se estima la elasticidad-precio global y por
categoría, y se entrena el modelo predictivo complementario.

## 4. Ejecutar la app de Streamlit
Desde la carpeta `app/` (o indicando la ruta completa):
```bat
cd app
streamlit run app_streamlit.py
```
Streamlit levantará un servidor local y abrirá el navegador automáticamente.
Si no se abre solo, copia la **Local URL** que aparece en Anaconda Prompt
(normalmente `http://localhost:8501`).

## 5. Qué contiene la app
- **Pestaña Exploración de datos**: curva de demanda, elasticidad por categoría,
  ranking de variantes más vendidas.
- **Pestaña Simulador de precio**: selecciona una pizza, mueve el slider de
  variación de precio (-30% a +30%) y observa el efecto estimado sobre
  cantidad demandada e ingreso, según la elasticidad calculada.

## Nota metodológica
El dataset no registra cambios de precio en el tiempo para un mismo producto
(el precio de cada variante es casi constante durante el año). Por eso la
elasticidad se estima de forma cruzada, comparando precio y cantidad **entre
variantes de pizza**, y se usa para *simular* escenarios de cambio de precio
— el mismo enfoque que utiliza Shuptar (2022) cuando señala que, al no existir
experimentos reales de precio, es necesario simular el efecto de una variación
porcentual arbitraria.
