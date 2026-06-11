# Proyecto Final - Minería de Datos Aplicada a México
El proyecto final de la materia de **Almacenes y Minería de Datos** tiene como objetivo, no solo aplicar los conocimientos 
clave adquiridos a lo largo del curso, sino además permitir el desarrollo de una solución completa de minería de datos 
orientada a un problema real de México.

Se selccionó el datset **Afluencia Diaria del Metro (Desglosada)**, 
disponible públicamente en [Portal de Datos Abiertos](https://datos.cdmx.gob.mx/dataset/afluencia-diaria-del-metro-cdmx/resource/cce544e1-dc6b-42b4-bc27-0d8e6eb3ed72), 
con el fin de desarrollar modelos de clasificación y clustering que permitan por un lado clasificar el nivel de saturación 
diario de cada estación del STC y por el otro segmentar las estaciones de acuerdo a sus perfiles de demanda.


## Equipo 2:
* **Escobar Gonzalez Isaac Giovani** - 321336400
* **Garduño Escobar Kevin Jonathan** - 321070629
* **Pinacho Báez Arlet** - 320287828
* **Sautto Ramirez Seldon** - 321084163

## Requerimientos:
Para la visualización de esta tarea se requiere tener instalado lo siguiente:
* Python 3.8 o superior (Si no lo tienes instalado, puedes descargarlo desde [aquí](https://www.python.org/downloads/))
* Quarto 1.9 o superior (Si no lo tienes instalado, puedes descargarlo desde [aquí](https://quarto.org/docs/download/))

## Instrucciones para ejecutar el código:
Seguir estos pasos para clonar el repositorio, crear el entorno virtual, instalar las dependencias y ejecutar el código:
1. Clonar el repositorio de GitHub
```bash
git clone https://github.com/arletpinacho/AyMD_proyecto.git
```
2. Navegar al directorio del proyecto
```bash
cd AyMD_proyecto
```
3. Crear un entorno virtual
```bash
python -m venv env
```
4. Activar el entorno virtual
- En Windows:
```bash
.\env\Scripts\activate
```
- En macOS/Linux:
```bash
source env/bin/activate
```
5. Instalar las dependencias
```bash
pip install -r requirements.txt
```
6. Crear la carpeta `data`
```bash
mkdir data
```
7. Colocar el archivo `afluenciastc_desglosado_04_2026` dentro de la carpeta `data/raw`
8. Navegar a la carpeta `notebooks/quarto`
```bash
cd notebooks/quarto
```
9. Ejecutar el código para la visualización
```bash
quarto render
```
También puedes ejecutar el código para la visualización utilizando el siguiente comando:
```bash
quarto preview
```
Esto abrirá una ventana en tu navegador con la visualización del análisis exploratorio de datos de personas desaparecidas en México.

## Instrucciones para ejecutar el script de demostración:
1. Asegúrate de haber seguido los pasos anteriores para configurar el entorno virtual e instalar las dependencias.
2. Deberás tener el modelo de clasificación previamente entrenado guardado en un archivo `.joblib` dentro de la carpeta `models`. Si no tienes el archivo, puedes ejecutar el notebook `clasificacion.qmd` para entrenar y guardar el modelo.
3. Desde el directorio raíz del proyecto, ejecuta el script de demostración con el siguiente comando:
```bash
python -m src.demostracion
```
Esto ejecutará el script `demostracion.py` e imprimirá en la consola los resultados de las predicciones realizadas por el modelo de clasificación.