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
