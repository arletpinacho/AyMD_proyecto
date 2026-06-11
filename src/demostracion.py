import sys
import os
import warnings
import pandas as pd

CARPETA_SRC = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(CARPETA_SRC, '..'))

if CARPETA_SRC not in sys.path:
    sys.path.insert(0, CARPETA_SRC)
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

from .modelo_clasificacion import ModeloClasificacion

warnings.filterwarnings('ignore')

def ejecutar_demostracion() -> None:
    """Carga el modelo persistido y clasifica cinco escenarios de ejemplo."""

    print("SISTEMA DE PREDICCIÓN DE SATURACIÓN — STC METRO CDMX")

    # 1. Localizar y cargar el modelo
    ruta_modelo = os.path.join(RAIZ, "models", "rf_clasificacion_metro.joblib")

    print(f"\n[1] Cargando modelo desde:")
    print(f"    {ruta_modelo}")

    if not os.path.exists(ruta_modelo):
        print("\n Archivo no encontrado.")
        print("Ejecuta clasificacion.qmd para entrenar y serializar el modelo.")
        return

    predictor = ModeloClasificacion.cargar_modelo(ruta_modelo, seed=42)

    if predictor is None:
        print("Error al deserializar el modelo.")
        return

    n_vars = len(predictor.feature_names_) if predictor.feature_names_ else "?"
    print(f"Pipeline cargado  — {n_vars} variables, "
          f"{predictor.pipeline_.named_steps['clasificador'].n_estimators} árboles")

    # 2. Construir escenarios de prueba
    print("\n[2] Escenarios de inferencia:")

    datos = pd.DataFrame({
        # Columnas en el mismo orden que X_train
        'linea_encoded'   : [0,    3,    10,   7,    2   ],
        'estacion_encoded': [114,  56,   142,  88,   33  ],
        'dia_semana'      : [0,    4,    6,    2,    1   ],  # 0=Lun … 6=Dom
        'mes_encoded'     : [3,    1,    5,    9,    2   ],  # orden alfabético
        'anio'            : [2025, 2024, 2024, 2023, 2025],
        'es_festivo'      : [0,    0,    0,    1,    1   ],
        'es_quincena'     : [1,    1,    0,    0,    1   ],
        'lluvia_historica': [0,    1,    1,    0,    0   ],
        'post_boleto'     : [1,    1,    1,    0,    1   ],
    })

    descripciones = [
        "L1 · Est.114 · Lunes   (Ene 2025) · Quincena",
        "L3 · Est.56  · Viernes (Ago 2024) · Lluvia",
        "LA · Est.142 · Domingo (Jul 2024) · Fin de mes · Lluvia",
        "L7 · Est.88  · Miérc. (Nov 2023) · Festivo (Revolución)",
        "L2 · Est.33  · Martes  (Dic 2025) · Festivo + Quincena",
    ]

    # 3. Inferencia
    print("\n[3] Ejecutando pipeline de inferencia...")

    predicciones   = predictor.predecir(datos)
    probabilidades = predictor.predecir_proba(datos)

    # 4. Presentar resultados
    print("RESULTADOS DE CLASIFICACIÓN")

    COLORES_NIVEL = {
        'Bajo':    '\033[92m',
        'Medio':   '\033[93m',
        'Alto':    '\033[33m',
        'Crítico': '\033[91m',
    }
    RESET = '\033[0m'

    print(f"\n  {'#':<3} {'Escenario':<50} {'Nivel':<10} {'Confianza'}")

    for i, (pred_idx, proba, desc) in enumerate(
            zip(predicciones, probabilidades, descripciones)):

        nivel     = ModeloClasificacion.NOMBRES_CLASES[pred_idx]
        confianza = proba.max() * 100
        color     = COLORES_NIVEL.get(nivel, '')

        print(f"  {i:<3} {desc:<50} "
              f"{color}{nivel:<10}{RESET} {confianza:>7.1f}%")

    print("\n  Distribución de probabilidades (Bajo / Medio / Alto / Crítico):")
    for i, (proba, desc) in enumerate(zip(probabilidades, descripciones)):
        dist = " | ".join(
            f"{ModeloClasificacion.NOMBRES_CLASES[j]}: {p*100:4.1f}%"
            for j, p in enumerate(proba)
        )
        print(f"  {i}: {dist}")

    print("Pipeline aplicado: StandardScaler → RandomForest")
    print("Los parámetros del scaler provienen del ajuste sobre X_train.")
    print("No se requirió re-entrenamiento.")


if __name__ == "__main__":
    ejecutar_demostracion()