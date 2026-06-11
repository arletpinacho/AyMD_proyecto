import pandas as pd
import holidays
from sklearn.preprocessing import LabelEncoder

def guardar_datos(df, ruta):
    df.to_csv(ruta, index=False)

def pivot_tipo_pago(df):
    df_simplificado = df.pivot_table(
        index=['fecha', 'linea', 'estacion', 'mes', 'anio', 'post_boleto'],
        columns='tipo_pago',
        values='afluencia',
        aggfunc='sum'
    ).reset_index()

    # renombramos las nuevas columnas
    df_simplificado.columns.name = None
    df_simplificado = df_simplificado.rename(columns={
        'Boleto': 'afluencia_boleto',
        'Prepago': 'afluencia_prepago',
        'Gratuidad': 'afluencia_gratuidad'
    })

    # calculamos la afluencia total
    df_simplificado['afluencia_total'] = (
        df_simplificado['afluencia_boleto'] + 
        df_simplificado['afluencia_prepago'] + 
        df_simplificado['afluencia_gratuidad']
    )

    return df_simplificado

def clasificar_saturacion(row):
    if row['afluencia_total'] <= row['p25']:
        return 0  # bajo
    elif row['afluencia_total'] <= row['p60']:
        return 1  # medio
    elif row['afluencia_total'] <= row['p85']:
        return 2  # alto
    else:
        return 3  # crítico

def agregar_variable_objetivo(df):
    percentiles = df.groupby(['linea', 'estacion'])['afluencia_total'].quantile(
        [0.25, 0.60, 0.85]
    ).unstack()
    percentiles.columns = ['p25', 'p60', 'p85']
    percentiles = percentiles.reset_index()
    df = df.merge(percentiles, on=['linea', 'estacion'], how='left')
    df['nivel_saturacion'] = df.apply(clasificar_saturacion, axis=1)
    return df

def codificar_categoricas(df):
    le_linea = LabelEncoder()
    le_mes = LabelEncoder()
    le_estacion = LabelEncoder()
    df['linea_encoded'] = le_linea.fit_transform(df['linea'])
    df['mes_encoded'] = le_mes.fit_transform(df['mes'])
    df['estacion_encoded'] = le_estacion.fit_transform(df['estacion'])
    return df, le_linea, le_mes, le_estacion

def agregar_vars_temporales(df):
    df['dia_semana'] = df['fecha'].dt.dayofweek
    anios = df['fecha'].dt.year.unique()
    mx_holidays = holidays.country_holidays('MX', years=anios)
    df['es_festivo'] = df['fecha'].isin(mx_holidays).astype(int)
    df['es_quincena'] = df['fecha'].dt.day.isin([14, 15, 30, 31]).astype(int)
    df['lluvia_historica'] = df['fecha'].dt.month.isin([5, 6, 7, 8, 9, 10]).astype(int)
    return df

def construir_df_clustering(df):
    df['proporcion_prepago'] = df['afluencia_prepago'] / df['afluencia_total']
    df['proporcion_gratuidad'] = df['afluencia_gratuidad'] / df['afluencia_total']
    df['proporcion_boleto'] = df['afluencia_boleto'] / df['afluencia_total']
    return df.groupby(['linea', 'estacion']).agg(
        afluencia_media=('afluencia_total', 'mean'),
        afluencia_std=('afluencia_total', 'std'),
        proporcion_prepago=('proporcion_prepago', 'mean'),
        proporcion_gratuidad=('proporcion_gratuidad', 'mean'),
        proporcion_boleto=('proporcion_boleto', 'mean')
    ).reset_index()