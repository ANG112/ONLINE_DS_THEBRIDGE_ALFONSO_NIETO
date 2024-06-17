
# Importaciones

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# Funciones
## eval_model: eval_model(target, predictions, problem_type, metrics);









########################################################################################################################################

def eval_model(target, predictions, problem_type, metrics):

    """
    Evalúa un modelo de regresión o clasificación en base a un conjunto de métricas especificadas.

    Parámetros:
    target : array tipo y_train
        Valores verdaderos de la target
    predictions : array tipo y_predict
        Valores predichos por el modelo.
    problem_type : str
        Tipo de problema ('regression' o 'classification').
    metrics : list of str
        Lista de métricas a calcular. 
        - Para regresión: ['RMSE', 'MAE', 'MAPE', 'GRAPH']
        - Para clasificación: ['ACCURACY', 'PRECISION', 'RECALL', 'CLASS_REPORT', 'MATRIX', 'MATRIX_RECALL', 'MATRIX_PRED', 'PRECISION_X', 'RECALL_X']
          donde 'X' es una etiqueta de alguna de las clases del target.

    Retorna:
    tuple
        Tupla con los resultados de las métricas calculadas en el orden especificado en la lista de métricas. 
        Las métricas que no devuelven valores numéricos retornan None en su lugar.

    Excepciones:
    ValueError
        Se lanza si ocurre un error en el cálculo de una métrica, si se especifica una métrica no soportada,
        Si el tipo de problema es inválido, o si se especifica una clase que no está presente en el target.

    Ejemplos de uso:
    >>> eval_model(y_true, y_pred, 'regression', ['RMSE', 'MAE', 'GRAPH'])
    >>> eval_model(y_true, y_pred, 'classification', ['ACCURACY', 'PRECISION', 'RECALL', 'MATRIX', 'PRECISION_1', 'RECALL_2'])
    """

    results = []
    
    if problem_type == "regression":
        for metric in metrics:
            if metric == "RMSE":
                rmse = np.sqrt(mean_squared_error(target, predictions))
                print(f"RMSE: {rmse}")
                results.append(rmse)
            elif metric == "MAE":
                mae = mean_absolute_error(target, predictions)
                print(f"MAE: {mae}")
                results.append(mae)
            elif metric == "MAPE":
                try:
                    mape = mean_absolute_percentage_error(target, predictions)
                    print(f"MAPE: {mape}")
                    results.append(mape)
                except Exception as e:
                    raise ValueError("Error calculating MAPE: " + str(e))
            elif metric == "GRAPH":
                plt.scatter(target, predictions)
                plt.xlabel("True Values")
                plt.ylabel("Predictions")
                plt.title("True vs Predicted Values")
                plt.show()
                results.append(None)
            else:
                raise ValueError(f"Unsupported regression metric: {metric}")

    elif problem_type == "classification":
        for metric in metrics:
            if metric == "ACCURACY":
                accuracy = accuracy_score(target, predictions)
                print(f"Accuracy: {accuracy}")
                results.append(accuracy)
            elif metric == "PRECISION":
                precision = precision_score(target, predictions, average='weighted')
                print(f"Precision: {precision}")
                results.append(precision)
            elif metric == "RECALL":
                recall = recall_score(target, predictions, average='weighted')
                print(f"Recall: {recall}")
                results.append(recall)
            elif metric == "CLASS_REPORT":
                report = classification_report(target, predictions)
                print("Classification Report:\n", report)
                results.append(None)
            elif metric == "MATRIX":
                matrix = confusion_matrix(target, predictions)
                disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
                disp.plot()
                plt.show()
                results.append(None)
            elif metric == "MATRIX_RECALL":
                matrix_recall = confusion_matrix(target, predictions, normalize='true')
                disp = ConfusionMatrixDisplay(confusion_matrix=matrix_recall)
                disp.plot()
                plt.show()
                results.append(None)
            elif metric == "MATRIX_PRED":
                matrix_pred = confusion_matrix(target, predictions, normalize='pred')
                disp = ConfusionMatrixDisplay(confusion_matrix=matrix_pred)
                disp.plot()
                plt.show()
                results.append(None)
            elif metric.startswith("PRECISION_"):
                class_label = metric.split("_")[1]
                if class_label not in np.unique(target):
                    raise ValueError(f"Class label {class_label} not found in target labels.")
                precision_class = precision_score(target, predictions, labels=[class_label], average='weighted')
                print(f"Precision for {class_label}: {precision_class}")
                results.append(precision_class)
            elif metric.startswith("RECALL_"):
                class_label = metric.split("_")[1]
                if class_label not in np.unique(target):
                    raise ValueError(f"Class label {class_label} not found in target labels.")
                recall_class = recall_score(target, predictions, labels=[class_label], average='weighted')
                print(f"Recall for {class_label}: {recall_class}")
                results.append(recall_class)
            else:
                raise ValueError(f"Unsupported classification metric: {metric}")

    else:
        raise ValueError("Invalid problem type. Use 'regression' or 'classification'.")

    return tuple(results)

