# TeamChallenge_team8

Explicación de las funciones:

1. describe_df
Función:

Genera un resumen de un DataFrame que incluye:
Tipos de datos de cada columna.
Porcentaje de valores faltantes.
Número de valores únicos.
Cardinalidad relativa (proporción de valores únicos respecto al total).
Argumentos:

dataframe: El DataFrame que se desea describir.
Salida:

Un nuevo DataFrame con las columnas:
DATA_TYPE: Tipo de dato de cada columna.
MISSINGS (%): Porcentaje de valores faltantes.
UNIQUE_VALUES: Cantidad de valores únicos.
CARDIN (%): Cardinalidad relativa.


2. tipifica_variables
Función:

Clasifica las columnas de un DataFrame en diferentes tipos según su cardinalidad y proporción de valores únicos.
Argumentos:

dataframe: El DataFrame que se desea analizar.
umbral_categoria: Umbral de cardinalidad para clasificar como categórica.
umbral_continua: Umbral de proporción para clasificar como continua.
Salida:

Un DataFrame con las columnas:
nombre_variable: Nombre de la columna.
tipo_sugerido: Clasificación sugerida:
Binaria: Si tiene solo 2 valores únicos.
Categorica: Si la cardinalidad es menor que umbral_categoria.
Numerica Continua: Si la proporción de valores únicos excede umbral_continua.
Numerica Discreta: En otros casos.


3. get_features_num_regression
Función:

Filtra las columnas numéricas que tienen una correlación significativa con una columna objetivo.
Argumentos:

dataframe: El DataFrame de entrada.
target_col: Columna objetivo (debe ser numérica).
umbral_corr: Umbral mínimo para la correlación absoluta.
pvalue (opcional): Valor 
𝑝
p para evaluar la significancia estadística (por defecto es None).
Salida:

Una lista con las columnas numéricas que cumplen con los criterios de correlación y significancia estadística.


4. plot_features_num_regression
Función:

Genera gráficos de correlación entre una columna objetivo y otras columnas numéricas seleccionadas.
Argumentos:

dataframe: El DataFrame de entrada.
target_col: Columna objetivo.
columns (opcional): Lista de columnas para graficar. Si no se especifica, utiliza todas las numéricas.
umbral_corr: Umbral de correlación mínima.
pvalue (opcional): Umbral para la significancia estadística.
Salida:

Gráficos de correlación (pairplots) de las columnas seleccionadas en relación con la columna objetivo.


5. get_features_cat_regression
Función:

Filtra las columnas categóricas que tienen una relación significativa con una columna objetivo numérica.
Argumentos:

dataframe: El DataFrame de entrada.
target_col: Columna objetivo (debe ser numérica).
pvalue: Umbral de significancia estadística.
Salida:

Una lista con las columnas categóricas que tienen una relación significativa con la columna objetivo.


6. plot_features_cat_regression
Función:

Genera gráficos para analizar la relación entre columnas categóricas y una columna objetivo numérica.
Argumentos:

dataframe: El DataFrame de entrada.
target_col: Columna objetivo.
columns (opcional): Lista de columnas categóricas para analizar.
pvalue: Umbral de significancia estadística.
with_individual_plot (opcional): Si es True, genera gráficos individuales para cada columna.
Salida:

Gráficos que muestran la relación entre las columnas categóricas y la columna objetivo. Si no hay columnas que cumplan los criterios, no se generarán gráficos.


Propósito general:
Estas funciones están diseñadas para:

Describir y clasificar las columnas de un DataFrame.
Filtrar columnas numéricas o categóricas en función de su relación con una columna objetivo.
Visualizar estas relaciones para facilitar el análisis exploratorio de datos (EDA) en problemas de regresión.
