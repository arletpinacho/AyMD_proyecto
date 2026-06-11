import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from .modelo_base import ModeloBase

warnings.filterwarnings('ignore')

class ModeloClustering(ModeloBase):
    """
    Implementación del modelo no supervisado para segmentar estaciones 
    basado en sus perfiles de afluencia y métodos de pago.
    Utiliza KMeans con Pipeline que encadena:
        - StandardScaler → normalización de features
        - KMeans         → clustering no supervisado

    Args:
        n_clusters (int): Número de clusters para el algoritmo de clustering. Por defecto es 4.
        seed (int): Semilla aleatoria para reproducibilidad. Por defecto es 42.
        metricas_validacion_ (dict): Diccionario para almacenar métricas de validación del clustering (inercia, silhouette).
    """

    NOMBRES_CLUSTERS = {
        0: "Alta Demanda Estructural",
        1: "Demanda Intermedia",
        2: "Baja Intensidad"
    }

    def __init__(self, n_clusters: int = 3, seed: int = 42):
        """
        Inicializa el modelo de clustering con una semilla fija para reproducibilidad y un número de clusters.
        
        Args:
            n_clusters (int): Número de clusters para el algoritmo de clustering. Por defecto es 3.
            seed (int): Semilla aleatoria para reproducibilidad. Por defecto es 42.
        """
        super().__init__(seed=seed)
        self.n_clusters = n_clusters
        self.metricas_validacion_ = None
        
    # Construcción del Pipeline
    def _construir_pipeline_base(self, n_clusters: int = None) -> Pipeline:
        """
        Construye el pipeline base con un escalador y un algoritmo de clustering KMeans.

        Args:
            n_clusters (int): Número de clusters para el algoritmo KMeans. Si no se proporciona, se utiliza el valor predeterminado del atributo n_clusters de la clase.

        Returns:
            Pipeline: Pipeline configurado con StandardScaler y KMeans.
        """
        k = n_clusters if n_clusters is not None else self.n_clusters
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clustering', KMeans(
                n_clusters=k,
                random_state=self.seed,
                n_init=10,
                max_iter=300
            ))
        ])
    
    def entrenar(self, X: pd.DataFrame, y=None, **kwargs) -> Dict:
        """
        Entrena el modelo de clustering con los datos proporcionados y calcula métricas de validación.
        Args:
            X (pd.DataFrame): Datos de entrenamiento para clustering.
            y: No se utiliza en modelos no supervisados, se ignora.
            **kwargs: Argumentos adicionales específicos del modelo (no utilizados en esta implementación).
        Returns:
            dict: Resultados del entrenamiento, incluyendo métricas de validación.
        """
        self.pipeline_ = self._construir_pipeline_base()
        self.feature_names_ = list(X.columns) if hasattr(X, 'columns') else None

        etiquetas_cluster = self.pipeline_.fit_predict(X)
        datos_escalados = self.pipeline_.named_steps['scaler'].transform(X)
        inercia = self.pipeline_.named_steps['clustering'].inertia_
        silueta = silhouette_score(datos_escalados, etiquetas_cluster)

        self.metricas_validacion_ = {
            'inercia': inercia,
            'silhouette_score': silueta
        }
        
        return {
            'estado': 'entrenado',
            'n_clusters': self.n_clusters,
            'inercia': inercia,
            'silhouette_score': silueta
        }

    def predecir(self, X: pd.DataFrame) -> np.ndarray:
        """
        Asigna etiquetas de cluster a los datos de entrada utilizando el modelo entrenado.

        Args:
            X (pd.DataFrame): Datos de entrada para asignar etiquetas de cluster.

        Returns:
            np.ndarray: Etiquetas de cluster asignadas a cada muestra en X.
        """
        return self.pipeline_.predict(X)

    def evaluar_k(self, X: pd.DataFrame, k_max: int = 8) -> Tuple[List[float], List[float]]:
        """
        Evalúa diferentes valores de k para el algoritmo KMeans utilizando las métricas de inercia y silhouette.
        Para cada valor de k se construye un pipeline temporal, se ajusta el modelo y se calculan las métricas correspondientes.

        Args:
            X (pd.DataFrame): Datos de entrada para evaluar los diferentes valores de k.
            k_max (int): Número máximo de clusters a evaluar. Por defecto es 8.

        Returns:
            Tuple[List[float], List[float]]: Listas con los valores de inercia y silhouette para cada k evaluado.
        """
        inercia = []
        silhouette = []
        for k in range(2, k_max + 1):
            pipeline_k = self._construir_pipeline_base(n_clusters=k)
            etiquetas_cluster = pipeline_k.fit_predict(X)

            datos_escalados = pipeline_k.named_steps['scaler'].transform(X)
            inercia.append(pipeline_k.named_steps['clustering'].inertia_)
            silhouette.append(silhouette_score(datos_escalados, etiquetas_cluster))
        
        return inercia, silhouette
    
    def Obtener_centroides(self) -> pd.DataFrame:
        """
        Obtiene los centroides de los clusters formados por el modelo entrenado.

        Returns:
            pd.DataFrame: DataFrame con los centroides de cada cluster, con columnas correspondientes a las características originales.

        Raises:
            ValueError: Si el modelo no ha sido entrenado y por lo tanto no se pueden obtener los centroides.
        """
        if self.pipeline_ is None:
            raise ValueError("El modelo no ha sido entrenado. No se pueden obtener los centroides.")
        
        kmeans = self.pipeline_.named_steps['clustering']
        scaler = self.pipeline_.named_steps['scaler']
        centroides_escalados = kmeans.cluster_centers_
        centroides_originales = scaler.inverse_transform(centroides_escalados)
        
        df_centroides = pd.DataFrame(centroides_originales, columns=self.feature_names_)
        df_centroides.index.name = 'Cluster'

        return df_centroides