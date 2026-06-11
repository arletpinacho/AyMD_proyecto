import pandas as pd
import unicodedata

def cargar_datos(ruta):
    return pd.read_csv(ruta, encoding="latin-1")

def reparar_acentos(texto):
    try:
        return texto.encode('raw_unicode_escape').decode('utf-8').encode('raw_unicode_escape').decode('utf-8')
    except Exception:
        try:
            return texto.encode('raw_unicode_escape').decode('utf-8')
        except Exception:
            return texto

# Función para quitar acentos de forma segura
def quitar_acentos(texto):
    if pd.isna(texto):
        return texto
    # Descompone los caracteres con acento y filtra dejando solo las letras base
    texto_normalizado = unicodedata.normalize('NFD', str(texto))
    return "".join(c for c in texto_normalizado if unicodedata.category(c) != 'Mn')

def limpiar_dataframe(df):
    df['linea'] = df['linea'].apply(lambda x: x.strip().split()[-1])
    df['estacion'] = df['estacion'].apply(reparar_acentos)
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['post_boleto'] = (df['fecha'] >= '2024-04-20').astype(int)
    return df

def normalizar_str(df):
    df['mes'] = df['mes'].str.upper()
    df['estacion'] = df['estacion'].str.upper().apply(quitar_acentos)
    return df

def filtrar_cierres(df):
    cierres = (
        # Línea 12: colapso tramo elevado
        ((df['linea'] == '12') & 
        (df['fecha'] >= '2021-05-03') & 
        (df['fecha'] <= '2024-01-30') & 
        (df['afluencia_total'] == 0)) |
        # Línea 1: remodelación
        ((df['linea'] == '1') & 
        (df['fecha'] >= '2022-07-09') & 
        (df['fecha'] <= '2023-11-16') & 
        (df['afluencia_total'] == 0))
    )

    return df[~cierres].copy()