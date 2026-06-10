import pandas as pd
import numpy as np
import time
import warnings
from typing import Optional, Dict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from .modelo_base import ModeloBase

warnings.filterwarnings('ignore')

class ModeloClasificacion(ModeloBase):
    """
    Implementación concreta del modelo de clasificación multiclase (4 niveles
    de saturación) del STC Metro CDMX.

    Utiliza Random Forest con Pipeline que encadena:
        - StandardScaler        → normalización de features
        - RandomForestClassifier → clasificación multiclase

    Args:
        busqueda_ (RandomizedSearchCV): Objeto de búsqueda de hiperparámetros.
        NOMBRES_CLASES (list): Lista de nombres de clases para interpretación de resultados.
        COLORES_CLASES (list): Paleta de colores para visualización.
    """
    
    NOMBRES_CLASES = ['Bajo', 'Medio', 'Alto', 'Crítico']
    COLORES_CLASES = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']

    def __init__(self, seed: int = 42):
        """Inicializa el modelo de clasificación con una semilla fija para reproducibilidad."""
        super().__init__(seed=seed)
        self.busqueda_ = None

    # Construcción del Pipeline
    def _construir_pipeline_base(self) -> Pipeline:
        """
        Construye el pipeline base con un escalador y un clasificador Random Forest.

        Returns:
            Pipeline: Pipeline configurado con StandardScaler y RandomForestClassifier.
        """
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clasificador', RandomForestClassifier(
                class_weight='balanced',
                random_state=self.seed,
                n_jobs=1
            ))
        ])
    
    def entrenar(self, X_train: pd.DataFrame, y_train: pd.Series, param_distributions: Dict = None, 
                n_iter: int = 10, n_splits: int = 3) -> Dict:
        """
        Entrena el modelo utilizando RandomizedSearchCV para optimizar los hiperparámetros.
        La métrica de optimización es 'f1_macro', que asigna igual peso a todas
        las clases, favoreciendo el desempeño en niveles menos frecuentes.

        Args:
            X_train (pd.DataFrame): Datos de entrenamiento (debe contener columnas numéricas).
            y_train (pd.Series): Etiquetas de entrenamiento (0=Bajo, 1=Medio, 2=Alto, 3=Crítico).
            param_distributions (Dict, optional): Diccionario de hiperparámetros. Si es None, usa defaults.
            n_iter (int): Número de iteraciones para RandomizedSearchCV.
            n_splits (int): Número de splits para la validación cruzada interna.

        Returns:
            Dict: Diccionario con el mejor score, mejores parámetros y tiempo de entrenamiento.
        """
        if param_distributions is None:
            param_distributions = {
                'clasificador__n_estimators': [150, 200, 300],
                'clasificador__max_depth': [20, 30, 40, None],
                'clasificador__min_samples_split': [2, 5],
                'clasificador__min_samples_leaf': [1, 2],
                'clasificador__max_features': ['sqrt', 'log2']
            }
        pipeline_base = self._construir_pipeline_base()
        cv_interna = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.seed
        )
        self.busqueda_ = RandomizedSearchCV(
            estimator=pipeline_base,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv_interna,
            scoring='f1_macro',
            n_jobs=None,
            random_state=self.seed,
            refit=True,
            return_train_score=False,
            verbose=2
        )
        t0 = time.time()
        self.busqueda_.fit(X_train, y_train)
        t1 = time.time()

        self.pipeline_      = self.busqueda_.best_estimator_
        self.feature_names_ = list(X_train.columns) if hasattr(X_train, 'columns') else None

        return {
            'mejor_score_f1': self.busqueda_.best_score_,
            'mejores_params': self.busqueda_.best_params_,
            'tiempo_seg'    : round(t1 - t0, 1)
        }
    
    def predecir(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones utilizando el modelo entrenado.
        
        Args:
            X (pd.DataFrame): Datos de entrada para realizar predicciones (debe contener columnas numéricas).

        Returns:
            np.ndarray: Predicciones generadas por el modelo (valores enteros 0-3 correspondientes a las clases).
        """
        return self.pipeline_.predict(X)

    def predecir_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza distribuciones de probabilidad sobre las 4 clases.
        
        Args:
            X (pd.DataFrame): Datos de entrada para realizar predicciones.

        Returns:
            np.ndarray: Probabilidades de pertenencia a cada clase para cada muestra.
        """
        return self.pipeline_.predict_proba(X)
    
    # Evaluación de rendimiento
    def evaluar(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Calcula el conjunto completo de métricas de evaluación sobre el conjunto de prueba.
        Métricas calculadas:
            - Accuracy, Precision, Recall, F1-Score.
            - Matriz de Confusión.
            - Reporte de Clasificación.
            - Predicciones probabilísticas

        Args:
            X_test (pd.DataFrame): Datos de prueba.
            y_test (pd.Series): Etiquetas verdaderas para los datos de prueba.

        Returns:
            Dict: Diccionario con métricas calculadas.
        """
        y_pred = self.predecir(X_test)
        y_proba = self.predecir_proba(X_test)
        return {
            'y_pred': y_pred,
            'y_proba': y_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_por_clase': precision_score(y_test, y_pred, average=None, labels=[0, 1, 2, 3]),
            'recall_por_clase': recall_score(y_test, y_pred, average=None, labels=[0, 1, 2, 3]),
            'f1_por_clase': f1_score(y_test, y_pred, average=None, labels=[0, 1, 2, 3]),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'precision_ponderada': precision_score(y_test, y_pred, average='weighted'),
            'recall_ponderado': recall_score(y_test, y_pred, average='weighted'),
            'f1_ponderado': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'report_texto': classification_report(y_test, y_pred, target_names=self.NOMBRES_CLASES)
        }
    
    # Análisis
    def importancia_variables(self) -> pd.DataFrame:
        """
        Extrae la importancia Gini de las variables del clasificador Random Forest.

        Returns:
            pd.DataFrame: DataFrame con nombres de variables e importancia, ordenado de mayor a menor.
        """
        if self.pipeline_ is None:
            raise ValueError("El modelo no ha sido entrenado.")

        rf = self.pipeline_.named_steps['clasificador']
        nombres = self.feature_names_ or [f'feature_{i}' for i in range(len(rf.feature_importances_))]
        return (
            pd.DataFrame({'Variable': nombres, 'Importancia': rf.feature_importances_})
            .sort_values('Importancia', ascending=False)
            .reset_index(drop=True)
        )
    
    def resultados_busqueda(self) -> pd.DataFrame:
        """
        Devuelve un DataFrame con los resultados de todas las combinaciones de hiperparámetros evaluadas en RandomizedSearchCV.

        Returns:
            pd.DataFrame: DataFrame con columnas de hiperparámetros y métricas de evaluación para cada combinación probada.
        """
        if self.busqueda_ is None:
            raise ValueError("El modelo no ha sido entrenado con búsqueda de hiperparámetros.")
        
        res = pd.DataFrame(self.busqueda_.cv_results_).sort_values(
            'mean_test_score', ascending=False
        )
        cols = [
            'param_clasificador__n_estimators',
            'param_clasificador__max_depth',
            'param_clasificador__min_samples_split',
            'param_clasificador__min_samples_leaf',
            'param_clasificador__max_features',
            'mean_test_score',
            'std_test_score'
        ]
        tabla = res[cols].copy()
        tabla.columns = ['n_est', 'max_d', 'min_spl', 'min_lf', 'max_ft', 'F1_macro', 'std']
        return tabla
    
    def baseline_dummy(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        Calcula la métrica de referencia utilizando un clasificador Dummy que siempre predice la clase más frecuente.

        Args:
            X_test (pd.DataFrame): Datos de prueba (solo se usarán las primeras 10 filas para entrenar el Dummy).
            y_test (pd.Series): Etiquetas verdaderas para los datos de prueba.

        Returns:
            float: Accuracy del clasificador Dummy en el conjunto de prueba.
        """
        dummy = DummyClassifier(strategy='most_frequent', random_state=self.seed)
        dummy.fit(X_test[:10], y_test[:10])
        return accuracy_score(y_test, dummy.predict(X_test))