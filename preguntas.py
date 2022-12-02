"""
Regresión Lineal Univariada
-----------------------------------------------------------------------------------------

En este laboratio se construirá un modelo de regresión lineal univariado.

"""
import numpy as np
import pandas as pd


def pregunta_01():
    """
    Carga de datos.
    -------------------------------------------------------------------------------------
    """
    # Lea el archivo `insurance.csv` y asignelo al DataFrame `df`
    df = pd.read_csv("insurance.csv")

    # Asigne la columna `charges` a la variable `y`.
    y = df["charges"].values

    # Asigne una copia del dataframe `df` a la variable `X`.
    X = df.copy()

    # Remueva la columna `charges` del DataFrame `X`.
    X = X.drop(columns="charges")

    # Retorne `X` y `y`
    return X, y


def pregunta_02():
    """
    Preparación de los conjuntos de datos.
    -------------------------------------------------------------------------------------
    """

    # Importe train_test_split
    from sklearn.model_selection import train_test_split

    # Cargue los datos y asigne los resultados a `X` y `y`.
    X, y = pregunta_01()

    # Divida los datos de entrenamiento y prueba. La semilla del generador de números
    # aleatorios es 12345. Use 300 patrones para la muestra de prueba.
    (X_train, X_test, y_train, y_test,) = train_test_split(
        X,
        y,
        test_size=300,
        random_state=12345,
    )

    # Retorne `X_train`, `X_test`, `y_train` y `y_test`
    return X_train, X_test, y_train, y_test


def pregunta_03():
    """
    Especificación del pipeline y entrenamiento
    -------------------------------------------------------------------------------------
    """

    # Importe make_column_selector
    # Importe make_column_transformer
    # Importe SelectKBest
    # Importe f_regression
    # Importe LinearRegression
    # Importe GridSearchCV
    # Importe Pipeline
    # Importe OneHotEncoder
    from sklearn.compose import make_column_selector
    from sklearn.compose import make_column_transformer
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_regression
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    pipeline = Pipeline(
        steps=[
            # Paso 1: Construya un column_transformer que aplica OneHotEncoder a las
            # variables categóricas, y no aplica ninguna transformación al resto de
            # las variables.
            (
                "column_transfomer",
                make_column_transformer(
                    (
                        OneHotEncoder(),
                        make_column_selector(dtype_include=object),
                    ),
                    remainder="passthrough"
                ),
            ),
            # Paso 2: Construya un selector de características que seleccione las K
            # características más importantes. Utilice la función f_regression.
            (
                "selectKBest",
                SelectKBest(score_func=f_regression, k=11),
            ),
            # Paso 3: Construya un modelo de regresión lineal.
            (
                "lr",
                LinearRegression(),
            ),
        ],
    )

    # Cargua de las variables.
    X_train, X_test, y_train, y_test = pregunta_02()

    # Defina un diccionario de parámetros para el GridSearchCV. Se deben
    # considerar valores desde 1 hasta 11 regresores para el modelo
    param_grid = {
      "lr__n_jobs" : list(range(1,12)),

  }

    # Defina una instancia de GridSearchCV con el pipeline y el diccionario de
    # parámetros. Use cv = 5, y como métrica de evaluación el valor negativo del
    # error cuadrático medio.
    gridSearchCV = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        refit=True,
        return_train_score=False,
    )

    # Búsque la mejor combinación de regresores
    gridSearchCV.fit(X_train, y_train)
    
    # Retorne el mejor modelo
    return gridSearchCV


def pregunta_04():
    """
    Particionamiento del conjunto de datos usando train_test_split.
    Complete el código presentado a continuación.
    """

    # Importe LinearRegression
    # Importe train_test_split
    # Importe mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv("gm_2008_region.csv")

    # Asigne a la variable los valores de la columna `fertility`
    X_fertility = df["fertility"].values.reshape(-1,1)

    # Asigne a la variable los valores de la columna `life`
    y_life = df["life"].values.reshape(-1,1)

    # Divida los datos de entrenamiento y prueba. La semilla del generador de números
    # aleatorios es 53. El tamaño de la muestra de entrenamiento es del 80%
    (X_train, X_test, y_train, y_test,) = train_test_split(
        X_fertility,
        y_life,
        test_size=0.2,
        random_state=53,
    )

    # Cree una instancia del modelo de regresión lineal
    LR = LinearRegression()

    # Entrene el clasificador usando X_train y y_train
    LR.fit(X_train, y_train)

    # Pronostique y_test usando X_test
    y_pred = LR.predict(X_test)

    # Compute and print R^2 and RMSE
    print("R^2: {:6.4f}".format(LR.score(X_test, y_test)))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Root Mean Squared Error: {:6.4f}".format(rmse))