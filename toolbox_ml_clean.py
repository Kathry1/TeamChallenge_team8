import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway, pearsonr


#1. describe df
def describe_df(dataframe):
    """
    Genera un resumen del DataFrame proporcionado. Devuelve un DataFrame con información sobre:
    - El tipo de dato de cada columna
    - El porcentaje de valores nulos
    - El número de valores únicos
    - La cardinalidad relativa de cada columna

    Argumentos:
    dataframe (pd.DataFrame): DataFrame de entrada para describir.

    Retorna:
    pd.DataFrame: Resumen con las métricas mencionadas para cada columna.
    """
    summary = {
        'COL_N': dataframe.columns,
        'DATA_TYPE': dataframe.dtypes,
        'MISSINGS (%)': dataframe.isnull().mean() * 100,
        'UNIQUE_VALUES': dataframe.nunique(),
        'CARDIN (%)': (dataframe.nunique() / len(dataframe)) * 100
    }

    summary_df = pd.DataFrame(summary)
    summary_df.set_index('COL_N', inplace=True)

    return summary_df

#2. tipifica variables
def tipifica_variables(dataframe, umbral_categoria, umbral_continua):

    # Clasifica las variables de un DataFrame en función de su cardinalidad y otros criterios.
    # Devuelve un DataFrame con dos columnas: "nombre_variable" y "tipo_sugerido".

    # Criterios:
    # - Si la cardinalidad es 2, asigna "Binaria".
    # - Si la cardinalidad es menor que umbral_categoria, asigna "Categorica".
    # - Si la cardinalidad es mayor o igual que umbral_categoria:
    # - Si el porcentaje de cardinalidad es mayor o igual a umbral_continua, asigna "Numerica Continua".
    # - En caso contrario, asigna "Numerica Discreta".

    # Argumentos:
    # dataframe (pd.DataFrame): DataFrame de entrada.
    # umbral_categoria (int): Umbral para determinar variables categoricas.
    # umbral_continua (float): Umbral para determinar variables continuas.

    # Retorna:
    # pd.DataFrame: DataFrame con "nombre_variable" y "tipo_sugerido".

    resultados = []
    for columna in dataframe.columns:
        cardinalidad = dataframe[columna].nunique()
        if cardinalidad == 2:
            tipo = "Binaria"
        elif cardinalidad < umbral_categoria:
            tipo = "Categorica"
        else:
            porcentaje_cardinalidad = cardinalidad / len(dataframe)
            if porcentaje_cardinalidad >= umbral_continua:
                tipo = "Numerica Continua"
            else:
                tipo = "Numerica Discreta"
        resultados.append({"nombre_variable": columna, "tipo_sugerido": tipo})

    return pd.DataFrame(resultados)


# 3.GET FEATURES NUM REGRESSION
def get_features_num_regression(dataframe, target_col, columns=None, umbral_corr=0, pvalue=None):

    # Genera pairplots para analizar la correlación entre una columna objetivo y otras columnas numéricas.

    # Argumentos:
    # dataframe (pd.DataFrame): DataFrame de entrada.
    # target_col (str): Columna objetivo para analizar la correlación.
    # columns (list, opcional): Lista de columnas para incluir en el análisis. Si es None, se seleccionan todas las numéricas.
    # umbral_corr (float, opcional): Umbral de correlación mínima (default = 0).
    # pvalue (float, opcional): Umbral de significancia estadística (default = None).

    # Retorna:
    # list: Lista de columnas que cumplen con los criterios y fueron incluidas en el gráfico.

    # Validación de entrada
    if target_col not in dataframe.columns:
        print(f"Error: La columna objetivo '{
            target_col}' no existe en el DataFrame.")
        return None

    if not pd.api.types.is_numeric_dtype(dataframe[target_col]):
        print("Error: La columna objetivo debe ser numérica.")
        return None

    if columns is None:
        columns = dataframe.select_dtypes(include=['number']).columns.tolist()
        columns.remove(target_col)

    if not columns:
        print("Error: No hay columnas numéricas para analizar.")
        return None

    filtered_columns = []
    for col in columns:
        if col in dataframe.columns:
            correlacion, p_valor = pearsonr(
                dataframe[col].dropna(), dataframe[target_col].dropna())
            if abs(correlacion) >= umbral_corr:
                if pvalue is None or p_valor < (1 - pvalue):
                    filtered_columns.append(col)

    if not filtered_columns:
        print("No hay columnas que cumplan con los criterios dados.")
        return None

    # Generar pairplot con seaborn
    max_columns = 5
    for i in range(0, len(filtered_columns), max_columns):
        subset = filtered_columns[i:i + max_columns] + [target_col]
        sns.pairplot(dataframe[subset])
        plt.show()

    return filtered_columns

# 4.
def plot_features_num_regression(dataframe, target_col, columns=None, umbral_corr=0, pvalue=None):

    # Genera pairplots para analizar la correlación entre una columna objetivo y otras columnas numéricas.

    # Argumentos:
    # dataframe (pd.DataFrame): DataFrame de entrada.
    # target_col (str): Columna objetivo para analizar la correlación.
    # columns (list, opcional): Lista de columnas para incluir en el análisis. Si es None, se seleccionan todas las numéricas.
    # umbral_corr (float, opcional): Umbral de correlación mínima (default = 0).
    # pvalue (float, opcional): Umbral de significancia estadística (default = None).

    # Retorna:
    # list: Lista de columnas que cumplen con los criterios y fueron incluidas en el gráfico.

    # Validación de entrada
    if target_col not in dataframe.columns:
        print(f"Error: La columna objetivo '{
            target_col}' no existe en el DataFrame.")
        return None

    if not pd.api.types.is_numeric_dtype(dataframe[target_col]):
        print("Error: La columna objetivo debe ser numérica.")
        return None

    if columns is None:
        columns = dataframe.select_dtypes(include=['number']).columns.tolist()
        columns.remove(target_col)

    if not columns:
        print("Error: No hay columnas numéricas para analizar.")
        return None

    filtered_columns = []
    for col in columns:
        if col in dataframe.columns:
            correlacion, p_valor = pearsonr(
                dataframe[col].dropna(), dataframe[target_col].dropna())
            if abs(correlacion) >= umbral_corr:
                if pvalue is None or p_valor < (1 - pvalue):
                    filtered_columns.append(col)

    if not filtered_columns:
        print("No hay columnas que cumplan con los criterios dados.")
        return None

    # Generar pairplot con seaborn
    max_columns = 5
    for i in range(0, len(filtered_columns), max_columns):
        subset = filtered_columns[i:i + max_columns] + [target_col]
        sns.pairplot(dataframe[subset])
        plt.show()

    return filtered_columns

# 5.
def get_features_cat_regression(dataframe, target_col, pvalue=0.05):

    # Filtra columnas categóricas de un DataFrame basándose en su relación estadística con una columna objetivo numérica.

    # Argumentos:
    # dataframe (pd.DataFrame): DataFrame de entrada.
    # target_col (str): Nombre de la columna objetivo (debe ser numérica continua o discreta con alta cardinalidad).
    # pvalue (float, opcional): Umbral de significancia estadística (default = 0.05).

    # Retorna:
    # list: Lista de columnas categóricas que cumplen con los criterios.
    # None: Si hay un error en los argumentos de entrada.

    # Validación de entrada
    if target_col not in dataframe.columns:
        print(f"Error: La columna objetivo '{
            target_col}' no existe en el DataFrame.")
        return None

    if not pd.api.types.is_numeric_dtype(dataframe[target_col]):
        print("Error: La columna objetivo debe ser numérica.")
        return None

    if not (0 <= pvalue <= 1):
        print("Error: El p-valor debe estar entre 0 y 1.")
        return None

    # Filtrar columnas categóricas
    columnas_categoricas = dataframe.select_dtypes(
        include=['object', 'category']).columns

    resultado = []

    for col in columnas_categoricas:
        if col in dataframe.columns:
            grupos = [dataframe[target_col][dataframe[col] == valor].dropna()
                      for valor in dataframe[col].unique()]
            if len(grupos) > 1:
                estadistico, p_valor = f_oneway(*grupos)
                if p_valor < pvalue:
                    resultado.append(col)

    return resultado


# 6.
def plot_features_cat_regression(dataframe, target_col, columns=None, pvalue=0.05, fill_na="Missing", with_individual_plot=False):
    """
    Analiza la relación entre variables categóricas y una variable target numérica.
    
    Args:
        dataframe (pd.DataFrame): DataFrame de entrada.
        target_col (str): Columna objetivo numérica.
        columns (list, optional): Columnas categóricas a analizar. Si es None, se seleccionan todas automáticamente.
        pvalue (float, optional): Umbral de significancia estadística para ANOVA (0.05 por defecto).
        fill_na (str, optional): Valor para reemplazar nulos en categóricas ("Missing" por defecto).
        with_individual_plot (bool, optional): Genera gráficos individuales si es True (False por defecto).

    Returns:
        selected_columns: Lista de columnas categóricas seleccionadas basadas en el análisis.
    """

    # Validaciones iniciales de la columna target
    if target_col not in dataframe.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no existe en el DataFrame.")
    if not pd.api.types.is_numeric_dtype(dataframe[target_col]):
        raise TypeError("La columna objetivo debe ser numérica.")

    # Selección automática de columnas categóricas si no se especifican
    if columns is None:
        columns = dataframe.select_dtypes(include=["object", "category"]).columns.tolist()

    if not columns:
        print("No hay columnas categóricas disponibles para analizar.")
        return []

    # Rellenar valores nulos con 'fill_na' en columnas categóricas
    dataframe = dataframe.copy()
    dataframe[columns] = dataframe[columns].fillna(fill_na)

    # Filtrar columnas significativas
    selected_columns = []
    for col in columns:
        unique_vals = dataframe[col].nunique()
        if unique_vals < 2:
            print(f"La columna '{col}' tiene menos de 2 categorías únicas. Ignorada.")
            continue
        
        # Se calcula el p_value de cada categórica frente al target para seleccionarla o dsecartarla
        grupos = [dataframe[target_col][dataframe[col] == valor].dropna() for valor in dataframe[col].unique()]
        if len(grupos) > 1:
            _, p_valor = f_oneway(*grupos)
            if p_valor < pvalue:
                selected_columns.append(col)

    if not selected_columns:
        print("No hay columnas categóricas que cumplan con el criterio de significancia.")
        return []

    # Generación de gráficos
    for col in selected_columns:
        plt.figure(figsize=(10, 6))
        if with_individual_plot:
            sns.boxplot(x=dataframe[col], y=dataframe[target_col])
            plt.title(f"Relación entre '{col}' y '{target_col}' (Boxplot)")
        else:
            sns.histplot(data=dataframe, x=target_col, hue=col, kde=True, element="step", stat="density")
            plt.title(f"Distribución de '{target_col}' por '{col}'")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return selected_columns