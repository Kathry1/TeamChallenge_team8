import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway, pearsonr


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

# Definir la función


def get_features_num_regression(dataframe, target_col, umbral_corr, pvalue=None):
    """
    Filtra columnas numéricas de un DataFrame basándose en su correlación con una columna objetivo y un p-valor opcional.

    Argumentos:
    dataframe (pd.DataFrame): DataFrame de entrada.
    target_col (str): Nombre de la columna objetivo (debe ser numérica continua o discreta con alta cardinalidad).
    umbral_corr (float): Umbral de correlación (entre 0 y 1).
    pvalue (float, opcional): Valor de p para filtrar columnas basadas en significancia estadística.

    Retorna:
    list: Lista de columnas numéricas que cumplen con los criterios.
    None: Si hay un error en los argumentos de entrada.
    """
    # Validación de entrada
    if target_col not in dataframe.columns:
        print(f"Error: La columna objetivo '{
              target_col}' no existe en el DataFrame.")
        return None

    if not (0 <= umbral_corr <= 1):
        print("Error: El umbral de correlación debe estar entre 0 y 1.")
        return None

    if pvalue is not None and not (0 <= pvalue <= 1):
        print("Error: El p-valor debe estar entre 0 y 1.")
        return None

    if not pd.api.types.is_numeric_dtype(dataframe[target_col]):
        print("Error: La columna objetivo debe ser numérica.")
        return None

    # Filtrar columnas numéricas
    columnas_numericas = dataframe.select_dtypes(include=['number']).columns
    columnas_numericas = [
        col for col in columnas_numericas if col != target_col]

    resultado = []

    for col in columnas_numericas:
        correlacion, p_valor = pearsonr(
            dataframe[col].dropna(), dataframe[target_col].dropna())
        if abs(correlacion) >= umbral_corr:
            if pvalue is None or p_valor < (1 - pvalue):
                resultado.append(col)

    return resultado


# Cargar el dataset
df = pd.read_csv(
    '/kaggle/input/sleep-health-and-lifestyle-dataset/Sleep_health_and_lifestyle_dataset.csv')

# Definir la columna objetivo y los parámetros
target_col = 'Sleep Duration'
umbral_corr = 0.3
pvalue = 0.05

# Aplicar la función
columnas_significativas = get_features_num_regression(
    df, target_col, umbral_corr, pvalue)

# Mostrar el resultado
print("Columnas significativas:", columnas_significativas)


# 3.GET FEATURES NUM REGRESSION
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


def plot_features_cat_regression(dataframe, target_col, columns=None, pvalue=0.05, with_individual_plot=False):

    # Genera gráficos de histogramas agrupados para analizar la relación entre variables categóricas y una columna objetivo numérica.

    # Argumentos:
    # dataframe (pd.DataFrame): DataFrame de entrada.
    # target_col (str): Columna objetivo para analizar la relación.
    # columns (list, opcional): Lista de columnas categóricas para incluir en el análisis. Si es None, se seleccionan todas las categóricas.
    # pvalue (float, opcional): Umbral de significancia estadística (default = 0.05).
    # with_individual_plot (bool, opcional): Si es True, genera gráficos individuales para cada columna.

    # Retorna:
    # list: Lista de columnas que cumplen con los criterios y fueron incluidas en los gráficos.

    # Validación de entrada
    if target_col not in dataframe.columns:
        print(f"Error: La columna objetivo '{
              target_col}' no existe en el DataFrame.")
        return None

    if not pd.api.types.is_numeric_dtype(dataframe[target_col]):
        print("Error: La columna objetivo debe ser numérica.")
        return None

    if columns is None:
        columns = dataframe.select_dtypes(
            include=['object', 'category']).columns.tolist()

    if not columns:
        print("Error: No hay columnas categóricas para analizar.")
        return None

    filtered_columns = []
    for col in columns:
        if col in dataframe.columns:
            grupos = [dataframe[target_col][dataframe[col] == valor].dropna()
                      for valor in dataframe[col].unique()]
            if len(grupos) > 1:
                estadistico, p_valor = f_oneway(*grupos)
                if p_valor < pvalue:
                    filtered_columns.append(col)

    if not filtered_columns:
        print("No hay columnas que cumplan con los criterios dados.")
        return None

    # Generar gráficos
    for col in filtered_columns:
        if with_individual_plot:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=dataframe[col], y=dataframe[target_col])
            plt.title(f"Relación entre {col} y {target_col}")
            plt.xticks(rotation=45)
            plt.show()

        else:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=dataframe, x=target_col, hue=col,
                         kde=True, element="step", stat="density")
            plt.title(f"Distribución de {target_col} por {col}")
            plt.xticks(rotation=45)
            plt.show()

    return filtered_columns
