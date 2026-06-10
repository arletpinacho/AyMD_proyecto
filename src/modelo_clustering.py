import pandas as pd
import numpy as np
import warnings
from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from .modelo_base import ModeloBase

warnings.filterwarnings('ignore')

class ModeloClustering(ModeloBase):
    """
    Implementación del modelo no supervisado para segmentar estaciones 
    basado en sus perfiles de afluencia y métodos de pago.
    """
    def __init__(self, n_clusters: int = 4, seed: int = 42):
        super().__init__(seed=seed)
        self.n_clusters = n_clusters
        self.metricas_validacion_ = None
        self.nombres_clusters_ = None
        
    def _construir_pipeline_base(self) -> Pipeline:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clustering', KMeans(
                n_clusters=self.n_clusters, 
                random_state=self.seed,
                n_init=10,
                max_iter=300
            )) #Si es que es Kmeans, si no, hay que cambiar
        ])
    
    def entrenar(self, X: pd.DataFrame, y=None, **kwargs) -> Dict:
        # Código de entrenamiento del modelo de clustering
        self.pipeline_ = self._construir_pipeline_base()
        self.pipeline_.fit(X)
        return self

    def predecir(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict(X)