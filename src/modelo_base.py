from abc import ABC, abstractmethod
import joblib
import os
from typing import Optional, Dict, Any

class ModeloBase(ABC):
    """
    Clase abstracta que define la interfaz base para todos los modelos de minería
    de datos del proyecto. 
    Implementa el Patrón Strategy: define un contrato común que permite
    intercambiar diferentes algoritmos sin cambiar el código cliente.

    Atributos:
        seed (int): Semilla aleatoria global para asegurar reproducibilidad.
        pipeline_ (Pipeline): Pipeline de procesamiento y modelo entrenado.
    """
    def __init__(self, seed: int = 42):
        """
        Inicializa la instancia con una semilla fija.

        Args:
            seed (int): Semilla aleatoria para reproducibilidad. Por defecto es 42.
        """
        self.seed = seed
        self.pipeline_ = None
        self.feature_names_ = None

    @abstractmethod
    def entrenar(self, X, y=None, **kwargs):
        """
        Entrena el modelo con los datos proporcionados.

        Args:
            X: Datos de entrenamiento (puede ser un DataFrame o matriz).
            y: Etiquetas de entrenamiento (opcional para modelos no supervisados).
            **kwargs: Argumentos adicionales específicos del modelo.

        Returns:
            dict: Resultados del entrenamiento.
        """
        pass

    @abstractmethod
    def predecir(self, X):
        """Realiza predicciones con el modelo entrenado.
        
        Args:
            X: Datos de entrada para realizar predicciones.

        Returns:
            Predicciones generadas por el modelo.
        """
        pass

    def guardar_modelo(self, ruta: str) -> str:
        """
        Serializa el pipeline entrenado en disco con joblib.

        Args:
            ruta (str): Ruta donde se guardará el modelo.
        
        Returns:
            str: Ruta donde se guardó el modelo.

        Raises:
            ValueError: Si el modelo no ha sido entrenado o el pipeline está vacío.
        """
        if self.pipeline_ is None:
            raise ValueError("El modelo no ha sido entrenado o el pipeline está vacío.")
            
        directorio = os.path.dirname(ruta)
        if directorio:
            os.makedirs(directorio, exist_ok=True)
            
        joblib.dump(self.pipeline_, ruta)
        return ruta

    @classmethod
    def cargar_modelo(cls, ruta: str, seed: int = 42) -> Optional['ModeloBase']:
        """
        Carga un modelo desde la ruta especificada.

        Args:
            ruta (str): Ruta del archivo del modelo a cargar.
            seed (int): Semilla aleatoria para reproducibilidad. Por defecto es 42.

        Returns:
            Optional[ModeloBase]: Instancia del modelo cargado o None si no se encuentra el archivo.
        """
        if not os.path.exists(ruta):
            print(f"Archivo no encontrado: {ruta}")
            return None
        try:
            instance = cls(seed=seed)
            instance.pipeline_ = joblib.load(ruta)
            print(f"Modelo cargado desde {ruta}")
            return instance
        except Exception as e:
            print(f"Error al cargar el modelo desde {ruta}: {e}")
            return None
        
    def obtener_info_modelo(self) -> Dict[str, Any]:
        """
        Devuelve información relevante sobre el modelo entrenado.

        Returns:
            dict: Información del modelo, incluyendo tipo de modelo, número de etapas en el pipeline y características utilizadas.
        """
        estado = "Entrenado" if self.pipeline_ is not None else "No entrenado"
        info = {
            'tipo_modelo': self.__class__.__name__,
            'estado': estado,
            'caracteristicas': self.feature_names_,
            "tipo_pipeline": type(self.pipeline_).__name__ if self.pipeline_ else None,
        }
        return info
    
    def __str__(self):
        info = self.obtener_info_modelo()
        return f"Modelo: {info['tipo_modelo']}, Estado: {info['estado']}, Características: {info['caracteristicas']}"