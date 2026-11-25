<div align="center">

# Procesamiento de Imágenes — Trabajo Práctico N°3

**Universidad Austral · Licenciatura en Ciencia de Datos · 2026**

**Análisis de videos · Detección de reposo y lectura de dados**

</div>

---

## 👥 Integrantes

| Apellido y nombre |
|---|
| Castillo, Juan Ignacio |
| Fasolato, Matías |
| Torregiani, Bautista |

---

## 📌 Introducción

Este repositorio contiene la resolución del **Trabajo Práctico N°3 de Procesamiento de Imágenes**. Se procesan cuatro videos de tiradas de **cinco dados rojos** para identificar cuándo quedan detenidos, leer los valores de sus caras superiores y generar videos con las detecciones etiquetadas.

| Punto | Objetivo |
|---|---|
| **A — Detección de reposo** | Encontrar automáticamente el tramo de frames en el que los cinco dados mantienen sus posiciones. |
| **B — Lectura de valores** | Contar los puntos visibles de cada dado e informar los valores por consola. |
| **C — Videos anotados** | Generar un MP4 por tirada con un rectángulo, un identificador y el valor de cada dado durante el tramo detectado. |

---

## 📁 Estructura del proyecto

```text
tp3-pdi-dados/
├── tirada_1.mp4                       # Videos originales de entrada
├── tirada_2.mp4
├── tirada_3.mp4
├── tirada_4.mp4
├── tp3_dados.py                       # Procesamiento y exportación de las cuatro tiradas
├── UA_LCD_PDI_TP3.pdf                # Consigna del trabajo práctico
├── resultados_tiradas/               # Carpeta generada al ejecutar el script
│   ├── figuras/                      # Imágenes y gráficos de las pruebas
│   │   ├── 01_movimiento_vs_reposo.png
│   │   ├── 03_diferencias_y_tramos.png
│   │   ├── 05_segmentacion_y_deteccion.png
│   │   ├── 06_umbrales_puntos.png
│   │   ├── 08_estabilidad_tirada_1.png
│   │   └── ...
│   ├── tirada_1_etiquetada.mp4        # Videos procesados
│   ├── tirada_2_etiquetada.mp4
│   ├── tirada_3_etiquetada.mp4
│   └── tirada_4_etiquetada.mp4
├── venv_pdi_tp3_dados/               # Entorno virtual local (ignorado por Git)
├── .gitignore
├── requirements.txt                  # Dependencias de Python
└── README.md
```

> **Nota:** `resultados_tiradas/` y `venv_pdi_tp3_dados/` se generan localmente. No es necesario incluirlos en el repositorio: se pueden recrear ejecutando el programa.

---

## ⚙️ Instalación

**1. Clonar el repositorio**

```bash
git clone https://github.com/TU_USUARIO/tp3-pdi-dados.git
cd tp3-pdi-dados
```

> Reemplazar `TU_USUARIO` por el usuario de GitHub donde está alojado el repositorio.

**2. Crear un entorno virtual**

```bash
python3 -m venv venv_pdi_tp3_dados
```

| Sistema | Comando de activación |
|---|---|
| Windows | `venv_pdi_tp3_dados\Scripts\activate` |
| Linux / macOS | `source venv_pdi_tp3_dados/bin/activate` |

**3. Instalar las dependencias**

```bash
python -m pip install -r requirements.txt
```

El proyecto utiliza **OpenCV**, **NumPy** y **Matplotlib**. El archivo `requirements.txt` debe incluir las tres dependencias.

---

## ▶️ Ejecución

> [!IMPORTANT]
> Ejecutar el script **desde la carpeta raíz del repositorio**, ya que los cuatro videos se cargan mediante rutas relativas.

```bash
python tp3_dados.py
```

- **Entrada:** `tirada_1.mp4` a `tirada_4.mp4`.
- **Procesamiento:** segmentación del color rojo en **HSV**, detección de dados mediante **componentes conectadas** y filtros de área y proporción; búsqueda del tramo de reposo por **comparación de centroides**; lectura de los puntos blancos mediante **umbralado** y componentes conectadas.
- **Salida:** tramos de reposo y valores por consola, gráficos y capturas de las pruebas en `resultados_tiradas/figuras/`, y cuatro videos etiquetados en `resultados_tiradas/`.

### 🔍 ¿Cómo se detecta el reposo?

El programa prueba distintos frames con cinco dados detectados como referencia. En los siguientes busca un candidato diferente cerca de cada centroide. Un frame cumple el criterio cuando los cinco centros están a **1 píxel o menos** de sus posiciones de referencia, medidas sobre la imagen reducida. Se selecciona el **tramo consecutivo más largo** de al menos **5 frames**. Un candidato extra no impide la asociación de los cinco dados.

Luego se toma el **frame central** del tramo para leer los puntos de cada dado. Para el conteo final se utiliza un **umbral de gris de 175** y un **área mínima de 5 píxeles** por punto, excluyendo las componentes que tocan el borde del recorte. Las anotaciones de ese frame se dibujan únicamente durante el tramo detectado; el video de salida conserva los demás frames.

### 🖼️ Visualizaciones

El script guarda imágenes de las etapas intermedias, como la comparación entre movimiento y reposo, las máscaras rojas, la detección de dados y la comparación de umbrales para contar puntos.

En `tp3_dados.py` se puede configurar:

```python
mostrar_imagenes = True  # False para guardar las figuras sin abrir ventanas
```

Con `True`, los gráficos también se muestran en pantalla y puede ser necesario cerrar cada ventana para continuar. Con `False`, los archivos se guardan igualmente.

---

## 📊 Resultados obtenidos

| Tirada | Tramo de reposo (frames) | Frame central | Valores D1 a D5 | Video de salida |
|---|---|---:|---|---|
| **1** | 67–84 | 75 | 4, 1, 3, 1, 2 | `tirada_1_etiquetada.mp4` |
| **2** | 64–71 | 67 | 3, 2, 1, 3, 2 | `tirada_2_etiquetada.mp4` |
| **3** | 65–85 | 75 | 2, 2, 4, 2, 2 | `tirada_3_etiquetada.mp4` |
| **4** | 65–87 | 76 | 2, 3, 5, 2, 2 | `tirada_4_etiquetada.mp4` |

Los videos procesados conservan la duración y el número de frames de los originales. El reposo se determina según la **estabilidad de las posiciones detectadas**, no mediante una medición directa de toda posible rotación del dado.
