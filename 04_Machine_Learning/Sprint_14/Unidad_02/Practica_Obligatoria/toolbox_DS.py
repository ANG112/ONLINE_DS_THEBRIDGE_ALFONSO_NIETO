''' 
Funciones contenidas:

- describe_df(dataframe)
- tipifica_variables(dataframe, umbral_categoria, umbral_continua)

- analisis_univariable_numericas(df,features_num)
- analisis_univariable_categoricas(df,features_cat)
- analisis_bivariable_categoricas_categorica(df,target,features_cat) # Target = categorica
- analisis_bivariable_numericas_categorica(df,target,features_num) # Target = categorica
- analisis_bivariable_numericas_numerica(df,target,features_num) # Target = numerica

- evaluar_mejor_modelo_regresion(X, y, modelos, cv=5, scoring='neg_mean_squared_error')
- evaluar_mejor_modelo_clasificacion(X, y, modelos, cv=5, scoring='accuracy')
- evaluar_mejor_modelo_scoring(X, y, modelos, cv=5, scorings=('accuracy',), verbose=True):

'''


# Importación de librerías

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_ind, f_oneway, pointbiserialr, chi2_contingency, shapiro, skew, kurtosis
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn import metrics
from sklearn import cluster
from sklearn.model_selection import cross_val_score
from sklearn.metrics import * 
from sklearn.metrics import accuracy_score, auc, average_precision_score, balanced_accuracy_score, calinski_harabasz_score, check_scoring, class_likelihood_ratios 
from sklearn.metrics import classification_report, cohen_kappa_score  
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans


### Funcion: describe_df 

def describe_df(dataframe):
    """
    Esta función analiza:
    - Los tipos de datos
    - Los valores faltantes
    - Los valores únicos
    - La cardinalidad
    de las variables de un DataFrame. 

    Argumentos:
    dataframe: DataFrame de Pandas

    Retorna:
    DataFrame donde las filas representan los tipos de datos, valores faltantes, etc.,
    y las columnas representan las variables del DataFrame.
    """

    # Lista de columnas del DataFrame
    lista_columnas = dataframe.columns.tolist()
    
    # Diccionario para almacenar los parámetros de cada columna
    diccionario_parametros = {}
    
    # Iteración sobre las columnas del DataFrame
    for columna in lista_columnas:
        # Tipo de datos de la columna
        tipo = dataframe[columna].dtype
        
        # Porcentaje de valores faltantes en la columna
        porc_nulos = round(dataframe[columna].isna().sum() / len(dataframe) * 100, 2)
        
        # Valores únicos en la columna
        valores_no_nulos = dataframe[columna].dropna()
        unicos = valores_no_nulos.nunique()
        
        # Cardinalidad de la columna
        cardinalidad = round(unicos / len(valores_no_nulos) * 100, 2)
        
        # Almacenar los parámetros de la columna en el diccionario
        diccionario_parametros[columna] = [tipo, porc_nulos, unicos, cardinalidad]
    
    # Construcción del DataFrame de resumen
    df_resumen = pd.DataFrame(diccionario_parametros, index=["DATE_TYPE", "MISSINGS (%)", "UNIQUE_VALUES", "CARDIN (%)"])
    
    # Retorno del DataFrame de resumen
    return df_resumen

#####################################################################################################################

### Funcion: tipifica_variables 

def tipifica_variables(dataframe, umbral_categoria, umbral_continua):
    """
    Esta función sugiere el tipo de variable para cada columna de un DataFrame
    basándose en la cardinalidad y umbrales proporcionados.

    Argumentos:
    dataframe: DataFrame de Pandas
    umbral_categoria: Entero, umbral para la cardinalidad que indica cuándo considerar una variable categórica.
    umbral_continua: Flotante, umbral para el porcentaje de cardinalidad que indica cuándo considerar una variable numérica continua.

    Retorna:
    DataFrame con dos columnas: "nombre_variable" y "tipo_sugerido",
    donde cada fila contiene el nombre de una columna del DataFrame y una sugerencia del tipo de variable.
    """

    # Lista para almacenar las sugerencias de tipos de variables
    sugerencias_tipos = []

    # Iteración sobre las columnas del DataFrame
    for columna in dataframe.columns:
        # Cálculo de la cardinalidad y porcentaje de cardinalidad
        cardinalidad = dataframe[columna].nunique()
        porcentaje_cardinalidad = (cardinalidad / len(dataframe)) * 100

        # Sugerencia del tipo de variable
        if cardinalidad == 2:
            tipo_sugerido = "Binaria"
        elif cardinalidad < umbral_categoria:
            tipo_sugerido = "Categórica"
        else:
            if porcentaje_cardinalidad >= umbral_continua:
                if dataframe[columna].dtype != 'object':
                    tipo_sugerido = "Numérica Continua"
                else:
                    tipo_sugerido = "Object"
            else:
                if dataframe[columna].dtype != 'object':
                    tipo_sugerido = "Numérica Discreta"
                else:
                    tipo_sugerido = "Object"
        # Agregar la sugerencia de tipo de variable a la lista
        sugerencias_tipos.append([columna, tipo_sugerido])

    # Construcción del DataFrame de sugerencias
    df_sugerencias = pd.DataFrame(sugerencias_tipos, columns=["nombre_variable", "tipo_sugerido"])

    # Retorno del DataFrame de sugerencias
    return df_sugerencias

#####################################################################################################################

def analisis_univariable_numericas(df, features_num):
    features_dist_normal = []
    features_no_dist_normal = []
    dict_features_num = {}

    for col in features_num:
        print(f'Para {col}')
        print('*'*25)

        # Se imprime el gráfico
        fig, axes = plt.subplots(figsize=(6, 4))
        sns.histplot(df[col], kde=True, ax=axes)
        axes.set_title(f'Histograma y KDE de {col}')
        plt.show()

        # Medidas de centralización, cuartiles y rangos
        valores_estadisticos = df[col].describe().round(2)
        print(valores_estadisticos)
        print()

        # Moda y coeficiente de variación
        moda = df[col].mode()[0]
        coef_variacion = (df[col].std() / df[col].mean()) * 100
        IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
        print(f'La moda es {moda:.2f}, el coeficiente de variación es {coef_variacion:.2f}% y el IQR es {IQR:.2f}')
        print()

        valores_anormales = 1.5 * IQR
        outliers = 3 * IQR

        # Se comprueba si tiene o no distribución normal con la prueba de Shapiro-Wilk
        res = round(shapiro(df[col])[1],4)
        asimetria, curtosis_val = None, None
        if res < 0.05:
            features_no_dist_normal.append(col)
            print(f'Prueba Shapiro-Wilk: p-value={res:.3f}. No tiene distribución normal')
            print()
            asimetria = round(skew(df[col]),2)
            if asimetria > 0.5:
                print(f'Asimetría: {asimetria:.2f}, valores extendidos a la derecha')
            elif asimetria < -0.5:
                print(f'Asimetría: {asimetria:.2f}, valores extendidos a la izquierda')
            else:
                print('Se puede considerar simétrica')
            print()
            
            curtosis_val = round(kurtosis(df[col]),2)
            if curtosis_val > 3:
                print(f'Curtosis: {curtosis_val:.2f}. Riesgo de valores atípicos')
            elif curtosis_val < 3:
                print(f'Curtosis: {curtosis_val:.2f}. Poco riesgo de valores atípicos')
            else:
                print(f'Curtosis: {curtosis_val:.2f}. Casi es una normal')
        else:
            features_dist_normal.append(col)
            print(f'Prueba Shapiro-Wilk: p-value={res:.3f}. Tiene distribución normal')

        # Outliers
        outliers_inferiores = (df[col] < (df[col].quantile(0.25) - outliers)).sum()
        outliers_inferiores_pro = (outliers_inferiores / len(df) * 100).round(2)
        outliers_superiores = (df[col] > (df[col].quantile(0.75) + outliers)).sum()
        outliers_superiores_pro = (outliers_superiores / len(df) * 100).round(2)
        print()
        print('Outliers:')
        print(f'Los outliers inferiores son {outliers_inferiores} y suponen en proporción {outliers_inferiores_pro}%')
        print()
        print(f'Los outliers superiores son {outliers_superiores} y suponen en proporción {outliers_superiores_pro}%')
        print()

        dict_features_num[col] = [res, asimetria, curtosis_val, outliers_inferiores_pro, outliers_superiores_pro]
    
    resumen_num = pd.DataFrame.from_dict(dict_features_num, orient='index', columns=['Shapiro', 'Asimetría', 'Curtosis', 'Outliers_inf_pro', 'Outliers_sup_pro'])

    return resumen_num

# Uso de la función
# df = pd.read_csv('path_to_your_dataset.csv')
# features_num = ['feature1', 'feature2', 'feature3']  # Lista de tus variables numéricas
# resumen_num = analisis_univariable_numericas(df, features_num)
# print(resumen_num)

#####################################################################################################################

def analisis_univariable_categoricas(df, features_cat):
    for col in features_cat:
        print(f'Para {col}')
        print('*'*25)

        # Conteo de frecuencias
        frecuencia = df[col].value_counts()
        porcentaje = df[col].value_counts(normalize=True) * 100

        # Se imprime el gráfico de barras
        fig, axes = plt.subplots(figsize=(8, 5))
        sns.countplot(x=df[col], ax=axes)
        axes.set_title(f'Distribución de {col}')
        plt.xticks(rotation=45)
        plt.show()

        # Resumen de estadísticas categóricas
        print(f'Frecuencia de categorías en {col}:')
        print(frecuencia)
        print()
        print(f'Porcentaje de categorías en {col}:')
        print(porcentaje.round(2))
        print()

        # Moda
        moda = df[col].mode()[0]
        print(f'La moda es {moda} con una frecuencia de {frecuencia[moda]}')
        print(f'Proporción de la moda: {porcentaje[moda]:.2f}%')
        print()
    
    return

# Uso de la función
# df = pd.read_csv('path_to_your_dataset.csv')
# features_cat = ['feature1', 'feature2', 'feature3']  # Lista de tus variables categóricas
# analisis_univariable_categoricas(df, features_cat)

#####################################################################################################################

def analisis_bivariable_categoricas_categorica(df, target, features_cat):
    for col in features_cat:
        if col == target:
            continue
        print(f'Análisis de {col} vs {target}')
        print('*' * 40)

        # Visualización con histogramas agrupados y KDE
        fig, axes = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, x=col, hue=target, ax=axes)
        axes.set_title(f'Distribución de {col} agrupada por {target}')
        plt.xticks(rotation=45)
        plt.show()

        # Tabla de contingencia con proporciones
        tabla_contingencia = pd.crosstab(df[col], df[target], normalize='index').round(2)
        print('Tabla de contingencia (proporciones):')
        print(tabla_contingencia)
        print()

        # Prueba chi-cuadrado
        tabla_contingencia_abs = pd.crosstab(df[col], df[target])
        chi2, p, dof, ex = chi2_contingency(tabla_contingencia_abs)
        print(f'Prueba Chi-cuadrado:')
        print(f'Chi-cuadrado: {chi2:.2f}')
        print(f'p-valor: {p:.3f}')
        print(f'Grados de libertad: {dof}')
        print()

        if p < 0.05:
            print(f'La variable {col} está significativamente asociada con {target} (p < 0.05). Podría ser útil para el modelo.')
        else:
            print(f'La variable {col} no está significativamente asociada con {target} (p >= 0.05).')
        print()

# Uso de la función
# df = pd.read_csv('path_to_your_dataset.csv')
# target = 'nombre_de_tu_variable_target'  # Tu variable objetivo
# features_cat = ['feature1', 'feature2', 'feature3']  # Lista de tus variables categóricas
# analisis_bivariable_categoricas(df, target, features_cat)

#####################################################################################################################

def analisis_bivariable_numericas_categorica(df, target, features_num):
    for col in features_num:
        print(f'Análisis de {col} vs {target}')
        print('*' * 40)

        # Visualización con gráfico de caja y violín
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.boxplot(x=target, y=col, data=df, ax=axes[0])
        axes[0].set_title(f'Gráfico de caja de {col} por {target}')
        
        sns.violinplot(x=target, y=col, data=df, ax=axes[1])
        axes[1].set_title(f'Gráfico de violín de {col} por {target}')
        plt.show()

        # Estadísticos de resumen
        resumen = df.groupby(target)[col].describe().round(2)
        print(f'Estadísticos de resumen para {col} por {target}:')
        print(resumen)
        print()

        # Prueba de hipótesis
        unique_values = df[target].nunique()
        if unique_values == 2:
            # Si la variable target es binaria
            categories = df[target].unique()
            group1 = df[df[target] == categories[0]][col]
            group2 = df[df[target] == categories[1]][col]
            t_stat, p_val = ttest_ind(group1, group2)
            print(f'Prueba t-student para {col}:')
            print(f'Estadístico t: {t_stat:.2f}')
            print(f'p-valor: {p_val:.3f}')
        else:
            # Si la variable target tiene más de dos categorías
            groups = [df[df[target] == cat][col] for cat in df[target].unique()]
            f_stat, p_val = f_oneway(*groups)
            print(f'ANOVA para {col}:')
            print(f'Estadístico F: {f_stat:.2f}')
            print(f'p-valor: {p_val:.3f}')
        
        if p_val < 0.05:
            print(f'La variable {col} está significativamente asociada con {target} (p < 0.05). Podría ser útil para el modelo.')
        else:
            print(f'La variable {col} no está significativamente asociada con {target} (p >= 0.05).')
        print()

# Uso de la función
# df = pd.read_csv('path_to_your_dataset.csv')
# target = 'nombre_de_tu_variable_target'  # Tu variable objetivo
# features_num = ['feature1', 'feature2', 'feature3']  # Lista de tus variables numéricas
# analisis_bivariable_numericas(df, target, features_num)


#####################################################################################################################

def analisis_bivariable_numericas_numerica(df, target, features_num):
    # Matriz de correlación con la variable objetivo
    correlaciones = abs(df[features_num].corr()[target]).sort_values(ascending=False)
    print("Matriz de correlación con la variable target:")
    print(correlaciones)
    print()

    # Heatmap de todas las variables numéricas
    corr_matrix = df[features_num + [target]].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Heatmap de Correlación')
    plt.show()

    # Variables que correlacionan por debajo y por encima del 20%
    low_corr_vars = corr_matrix[target][(abs(corr_matrix[target]) < 0.2)].index.tolist()
    high_corr_vars = corr_matrix[target][(abs(corr_matrix[target]) >= 0.2)].index.tolist()
    print(f"Variables con correlación por debajo del 20%: {low_corr_vars}")
    print(f"Variables con correlación por encima del 20%: {high_corr_vars}")
    print()

    significant_vars = []

    for col in features_num:
        print(f'Prueba de correlación para {col} vs {target}')
        print('*' * 40)

        # Selección de prueba de correlación
        if df[col].dtype in ['float64', 'int64']:
            if df[col].nunique() > 20:
                corr_test, p_val = pearsonr(df[col], df[target])
                corr_type = 'Pearson'
            else:
                corr_test, p_val = spearmanr(df[col], df[target])
                corr_type = 'Spearman'
        else:
            corr_test, p_val = kendalltau(df[col], df[target])
            corr_type = 'Kendall'

        print(f'Prueba de correlación ({corr_type}):')
        print(f'Estadístico de correlación: {corr_test:.2f}')
        print(f'p-valor: {p_val:.3f}')

        if p_val < 0.05 and abs(corr_test) >= 0.2:
            print(f'La variable {col} está significativamente asociada con {target} (p < 0.05 y correlación >= 0.2). Podría ser útil para el modelo.')
            significant_vars.append(col)
        else:
            print(f'La variable {col} no está significativamente asociada con {target} (p >= 0.05 o correlación < 0.2).')
        print()

    print(f'Variables significativas para el modelo: {significant_vars}')
    print()

    # Análisis de multicolinealidad para variables significativas (excluyendo target)
    if significant_vars:
        significant_vars_ex_target = [var for var in significant_vars if var != target]
        if significant_vars_ex_target:
            print('Análisis de multicolinealidad:')
            X = df[significant_vars_ex_target]
            vif_data = pd.DataFrame()
            vif_data["Feature"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
            print(vif_data)
            print()

            high_vif_vars = vif_data[vif_data["VIF"] > 5]["Feature"].tolist()
            if high_vif_vars:
                print(f'Variables con VIF por encima de 5: {high_vif_vars}')
            else:
                print('No hay variables con VIF por encima de 5.')
        else:
            print('No hay variables significativas para el análisis de multicolinealidad.')
    else:
        print('No hay variables significativas para el análisis de multicolinealidad.')

# Uso de la función
# df = pd.read_csv('path_to_your_dataset.csv')
# target = 'nombre_de_tu_variable_target'  # Tu variable objetivo
# features_num = ['feature1', 'feature2', 'feature3']  # Lista de tus variables numéricas
# analisis_bivariable_numericas_numerica(df, target, features_num)


#####################################################################################################################

def evaluar_mejor_modelo_regresion(X, y, modelos, cv=5, scoring='neg_mean_squared_error'):
    """
    Evalúa el mejor modelo de regresión utilizando cross_val_score.

    Parámetros:
    X : pandas DataFrame o numpy array
        Matriz de características.
    y : pandas Series o numpy array
        Vector objetivo.
    modelos : dict
        Diccionario con los nombres de los modelos como claves y los objetos de los modelos como valores.
    cv : int, opcional (por defecto=5)
        Número de particiones para la validación cruzada.
    scoring : str, opcional (por defecto='neg_mean_squared_error')
        Métrica de evaluación, debe ser una de las métricas permitidas por scikit-learn.

    Retorna:
    None
    """
   

    resultados = {}
    
    for nombre, modelo in modelos.items():
        print(f'Evaluando modelo: {nombre}')
        scores = cross_val_score(modelo, X, y, cv=cv, scoring=scoring)
        media_scores = np.mean(scores)
        resultados[nombre] = media_scores
        print(f'{nombre} - Puntaje medio de {scoring}: {media_scores:.4f}')
    
    # Seleccionar el mejor modelo
    mejor_modelo = max(resultados, key=resultados.get)
    mejor_puntaje = resultados[mejor_modelo]
    
    print('\nResultados de todos los modelos:')
    for nombre, puntuacion in resultados.items():
        print(f'{nombre}: {puntuacion:.4f}')
    
    print(f'\nEl mejor modelo es {mejor_modelo} con un puntaje medio de {mejor_puntaje:.4f}')

# Ejemplo de uso
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR

# modelos = {
#     'Regresión Lineal': LinearRegression(),
#     'Árbol de Decisión': DecisionTreeRegressor(),
#     'Bosque Aleatorio': RandomForestRegressor(),
#     'Máquinas de Vectores de Soporte': SVR()
# }

# X = ...  # Matriz de características
# y = ...  # Vector objetivo

# evaluar_mejor_modelo_regresion(X, y, modelos, cv=5, scoring='neg_mean_squared_error')


#####################################################################################################################

def evaluar_mejor_modelo_clasificacion(X, y, modelos, cv=5, scoring='accuracy'):
    """
    Evalúa el mejor modelo de clasificación utilizando cross_val_score.

    Parámetros:
    X : pandas DataFrame o numpy array
        Matriz de características.
    y : pandas Series o numpy array
        Vector objetivo.
    modelos : dict
        Diccionario con los nombres de los modelos como claves y los objetos de los modelos como valores.
    cv : int, opcional (por defecto=5)
        Número de particiones para la validación cruzada.
    scoring : str, opcional (por defecto='accuracy')
        Métrica de evaluación, debe ser una de las métricas permitidas por scikit-learn.

    Retorna:
    None
    """
  

    resultados = {}
    
    for nombre, modelo in modelos.items():
        print(f'Evaluando modelo: {nombre}')
        scores = cross_val_score(modelo, X, y, cv=cv, scoring=scoring)
        media_scores = np.mean(scores)
        resultados[nombre] = media_scores
        print(f'{nombre} - Puntaje medio de {scoring}: {media_scores:.4f}')
    
    # Seleccionar el mejor modelo
    mejor_modelo = max(resultados, key=resultados.get)
    mejor_puntaje = resultados[mejor_modelo]
    
    print('\nResultados de todos los modelos:')
    for nombre, puntuacion in resultados.items():
        print(f'{nombre}: {puntuacion:.4f}')
    
    print(f'\nEl mejor modelo es {mejor_modelo} con un puntaje medio de {mejor_puntaje:.4f}')

# Ejemplo de uso
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC

# modelos = {
#     'Regresión Logística': LogisticRegression(),
#     'Árbol de Decisión': DecisionTreeClassifier(),
#     'Bosque Aleatorio': RandomForestClassifier(),
#     'Máquinas de Vectores de Soporte': SVC()
# }

# X = ...  # Matriz de características
# y = ...  # Vector objetivo

# evaluar_mejor_modelo_clasificacion(X, y, modelos, cv=5, scoring='accuracy')

#####################################################################################################################

def evaluar_mejor_modelo_scoring(X, y, modelos, cv=5, scorings=('accuracy',), verbose=True):
    """
    Evalúa el mejor modelo de clasificación utilizando cross_val_score.

    Parámetros:
    X : pandas DataFrame o numpy array
        Matriz de características.
    y : pandas Series o numpy array
        Vector objetivo.
    modelos : dict
        Diccionario con los nombres de los modelos como claves y los objetos de los modelos como valores.
    cv : int, opcional (por defecto=5)
        Número de particiones para la validación cruzada.
    scorings : tuple de str, opcional (por defecto=('accuracy',))
        Métricas de evaluación, debe ser una o hasta tres de las métricas permitidas por scikit-learn.
    verbose : bool, opcional (por defecto=True)
        Controla si mostrar o no la información detallada.

    Retorna:
    None
    """
    

    resultados = {}
    mejores_resultados = {}
    
    for nombre, modelo in modelos.items():
        if verbose:
            print(f'Evaluando modelo: {nombre}')
        scores = {}
        for scoring in scorings:
            scores[scoring] = np.mean(cross_val_score(modelo, X, y, cv=cv, scoring=scoring))
            if verbose:
                print(f'{nombre} - Puntaje medio de {scoring}: {scores[scoring]:.4f}')
        resultados[nombre] = scores
    
    # Seleccionar el mejor modelo y scoring
    mejor_modelo = None
    mejor_puntaje = float('-inf')
    mejor_scoring = None
    
    for nombre, scores in resultados.items():
        for scoring, puntaje in scores.items():
            if puntaje > mejor_puntaje:
                mejor_modelo = nombre
                mejor_puntaje = puntaje
                mejor_scoring = scoring
    
    if verbose:
        print('\nResultados de todos los modelos:')
        for nombre, scores in resultados.items():
            print(f'{nombre}: {scores}')

        print(f'\nEl mejor modelo es {mejor_modelo} con un puntaje medio de {mejor_puntaje:.4f} utilizando {mejor_scoring} como scoring')

    return mejor_modelo, mejor_scoring


#####################################################################################################################



