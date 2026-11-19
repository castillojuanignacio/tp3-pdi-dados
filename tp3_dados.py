import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- Configuración general ---

mostrar_imagenes = True  # False para correr sin abrir ventanas

carpeta_salida = "resultados_tiradas"
carpeta_figuras = os.path.join(carpeta_salida, "figuras")
os.makedirs(carpeta_figuras, exist_ok=True)


def mostrar_paneles(imagenes, titulos, titulo_figura, nombre_archivo):
    """Muestra (y guarda en resultados_tiradas/figuras) varias imágenes
    lado a lado. Acepta imágenes BGR (color) o de un solo canal."""
    fig, ejes = plt.subplots(1, len(imagenes), figsize=(5 * len(imagenes), 5))
    ejes = np.atleast_1d(ejes)

    for eje, imagen, titulo in zip(ejes, imagenes, titulos):
        if imagen.ndim == 2:
            eje.imshow(imagen, cmap="gray")
        else:
            eje.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        eje.set_title(titulo)
        eje.axis("off")

    fig.suptitle(titulo_figura)
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta_figuras, nombre_archivo), dpi=120)

    if mostrar_imagenes:
        plt.show()
    plt.close(fig)


def dibujar_anotaciones(frame, anotaciones, grosor=2, escala_texto=0.45):
    """Devuelve una copia del frame con las cajas y etiquetas dibujadas."""
    copia = frame.copy()
    for x, y, ancho, alto, texto in anotaciones:
        cv2.rectangle(copia, (x, y), (x + ancho - 1, y + alto - 1),
                      (0, 255, 0), grosor)
        cv2.putText(copia, texto, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, escala_texto, (0, 255, 0), 1)
    return copia


# --- Punto A: diferencia entre frames consecutivos ---

cap = cv2.VideoCapture("tirada_1.mp4")

if not cap.isOpened():
    raise SystemExit("No se pudo abrir tirada_1.mp4")

gris_anterior = None
numero_frame = 0
diferencias_por_frame = []
paneles_prueba_1 = []  # (frame, diferencia, título) de los frames 45 y 70

while True:
    ret, frame = cap.read()
    if not ret:
        break

    alto, ancho = frame.shape[:2]
    frame_chico = cv2.resize(frame, (ancho // 3, alto // 3))
    gris_actual = cv2.cvtColor(frame_chico, cv2.COLOR_BGR2GRAY)

    if gris_anterior is not None:
        diferencia = cv2.absdiff(gris_actual, gris_anterior)
        promedio = np.mean(diferencia)
        diferencias_por_frame.append((numero_frame, promedio))

        # --- Prueba 1: comparar movimiento y reposo ---
        # 44 -> 45: dados en movimiento. En la Mac dio 2.825.
        # 69 -> 70: dados quietos. En la Mac dio 0.124.
        if numero_frame in (45, 70):
            print(f"Frames {numero_frame - 1} y {numero_frame}: {promedio:.3f}")

            paneles_prueba_1.append(
                (frame_chico.copy(), diferencia.copy(),
                 f"Frames {numero_frame - 1}→{numero_frame} (dif. media {promedio:.3f})")
            )

    # --- Prueba 4: segmentar los dados rojos en el frame 70 ---
    if numero_frame == 70:
        frame_hsv = cv2.cvtColor(frame_chico, cv2.COLOR_BGR2HSV)

        rojo_bajo = cv2.inRange(frame_hsv, (0, 120, 60), (10, 255, 255))
        rojo_alto = cv2.inRange(frame_hsv, (170, 120, 60), (179, 255, 255))
        mascara_roja = np.uint8(np.logical_or(rojo_bajo, rojo_alto)) * 255

        guardado_frame = cv2.imwrite(os.path.join(carpeta_figuras, "prueba_frame_70.png"), frame_chico)
        guardada_mascara = cv2.imwrite(os.path.join(carpeta_figuras, "prueba_mascara_roja_70.png"), mascara_roja)
        print(f"Frame guardado: {guardado_frame} | Máscara guardada: {guardada_mascara}")

    gris_anterior = gris_actual
    numero_frame += 1

cap.release()

print(f"\nFrames leídos: {numero_frame}")
numero_frame_total = numero_frame

# Figura de la Prueba 1: dados en movimiento vs. dados quietos
if len(paneles_prueba_1) == 2:
    mostrar_paneles(
        [paneles_prueba_1[0][0], paneles_prueba_1[0][1],
         paneles_prueba_1[1][0], paneles_prueba_1[1][1]],
        ["Frame 45 (movimiento)", paneles_prueba_1[0][2],
         "Frame 70 (reposo)", paneles_prueba_1[1][2]],
        "Prueba 1: diferencia entre frames consecutivos",
        "01_movimiento_vs_reposo.png",
    )

# --- Prueba 2: comparar umbrales sobre el video completo ---

umbrales = [0.20, 0.30, 0.40, 0.50, 0.60, 1.00]
umbral_inicial = 0.30
candidatos_umbral_inicial = []

for umbral in umbrales:
    frames_candidatos = []

    for numero_frame, promedio in diferencias_por_frame:
        if promedio < umbral:
            frames_candidatos.append(numero_frame)

    print(f"Umbral {umbral:.2f}: {len(frames_candidatos)} frames candidatos")

    if umbral == umbral_inicial:
        candidatos_umbral_inicial = frames_candidatos

# --- Prueba 3: agrupar frames candidatos consecutivos ---

minimo_consecutivos = 5
tramos = []

for numero_frame in candidatos_umbral_inicial:
    if tramos and numero_frame == tramos[-1][1] + 1:
        tramos[-1][1] = numero_frame
    else:
        tramos.append([numero_frame, numero_frame])

for inicio, fin in tramos:
    cantidad = fin - inicio + 1

    if cantidad >= minimo_consecutivos:
        print(f"Tramo candidato: frames {inicio} a {fin} ({cantidad} frames)")

# Gráfico de la diferencia media por frame con el umbral y los tramos
fig, eje = plt.subplots(figsize=(10, 4))
eje.plot([f for f, _ in diferencias_por_frame],
         [d for _, d in diferencias_por_frame], label="Diferencia media")
eje.axhline(umbral_inicial, color="r", linestyle="--",
            label=f"Umbral {umbral_inicial}")
for inicio, fin in tramos:
    if fin - inicio + 1 >= minimo_consecutivos:
        eje.axvspan(inicio, fin, color="green", alpha=0.25)
eje.set_xlabel("Frame")
eje.set_ylabel("Diferencia media (niveles de gris)")
eje.set_title("Prueba 3: tramos con dados en reposo (tirada_1)")
eje.legend()
fig.tight_layout()
fig.savefig(os.path.join(carpeta_figuras, "03_diferencias_y_tramos.png"), dpi=120)
if mostrar_imagenes:
    plt.show()
plt.close(fig)

# --- Prueba 5: detectar los dados en la máscara del frame 70 ---

mascara_70 = cv2.imread(os.path.join(carpeta_figuras, "prueba_mascara_roja_70.png"), cv2.IMREAD_GRAYSCALE)
frame_con_cajas = cv2.imread(os.path.join(carpeta_figuras, "prueba_frame_70.png"))

if mascara_70 is None or frame_con_cajas is None:
    raise SystemExit("No se pudieron leer los PNG de la Prueba 4")

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    mascara_70, connectivity=8
)

cantidad_dados = 0

# La etiqueta 0 corresponde al fondo.
for etiqueta in range(1, num_labels):
    x, y, ancho, alto, area = stats[etiqueta]
    proporcion = ancho / alto

    print(f"Componente {etiqueta}: {ancho} x {alto}, área={area}, "
          f"ancho/alto={proporcion:.2f}")

    if area >= 100 and 0.5 <= proporcion <= 2.0:
        cantidad_dados += 1
        cv2.rectangle(
            frame_con_cajas,
            (x, y),
            (x + ancho - 1, y + alto - 1),
            (0, 255, 0),
            2
        )

print(f"Dados candidatos en el frame 70: {cantidad_dados}")
cv2.imwrite(os.path.join(carpeta_figuras, "prueba_dados_detectados_70.png"), frame_con_cajas)

mostrar_paneles(
    [cv2.imread(os.path.join(carpeta_figuras, "prueba_frame_70.png")), mascara_70, frame_con_cajas],
    ["Frame 70", "Máscara roja (HSV)", f"Dados detectados: {cantidad_dados}"],
    "Pruebas 4 y 5: segmentación y detección de los dados",
    "05_segmentacion_y_deteccion.png",
)

# --- Prueba 6: comparar umbrales en un solo dado ---

def contar_puntos(dado, umbral, area_minima=3):
    alto, ancho = dado.shape[:2]
    dado_gris = cv2.cvtColor(dado, cv2.COLOR_BGR2GRAY)
    _, mascara_puntos = cv2.threshold(
        dado_gris, umbral, 255, cv2.THRESH_BINARY
    )

    num_componentes, _, stats_puntos, _ = cv2.connectedComponentsWithStats(
        mascara_puntos, connectivity=8
    )

    cantidad_puntos = 0

    for etiqueta in range(1, num_componentes):
        x, y, ancho_punto, alto_punto, area = stats_puntos[etiqueta]

        toca_borde = (
            x == 0 or y == 0
            or x + ancho_punto == ancho
            or y + alto_punto == alto
        )

        if area >= area_minima and not toca_borde:
            cantidad_puntos += 1

    return cantidad_puntos, mascara_puntos


frame_original_70 = cv2.imread(os.path.join(carpeta_figuras, "prueba_frame_70.png"))

x, y, ancho, alto, _ = stats[1]
dado_de_prueba = frame_original_70[y:y + alto, x:x + ancho]

paneles_umbrales = [dado_de_prueba]
titulos_umbrales = ["Dado recortado"]

for umbral in [100, 160, 220]:
    cantidad, mascara = contar_puntos(dado_de_prueba, umbral)
    print(f"Umbral {umbral}: {cantidad} puntos candidatos")
    cv2.imwrite(os.path.join(carpeta_figuras, f"prueba_puntos_umbral_{umbral}.png"), mascara)
    paneles_umbrales.append(mascara)
    titulos_umbrales.append(f"Umbral {umbral}: {cantidad} puntos")

mostrar_paneles(
    paneles_umbrales, titulos_umbrales,
    "Prueba 6: umbral para segmentar los puntos de un dado",
    "06_umbrales_puntos.png",
)

# --- Prueba 7: leer y etiquetar los cinco dados ---

umbral_puntos_elegido = 160
frame_etiquetado = frame_con_cajas.copy()
id_dado = 0
anotaciones_70 = []

for etiqueta in range(1, num_labels):
    x, y, ancho, alto, area = stats[etiqueta]

    if area >= 100 and 0.5 <= ancho / alto <= 2.0:
        dado = frame_original_70[y:y + alto, x:x + ancho]
        valor, _ = contar_puntos(dado, umbral_puntos_elegido)

        id_dado += 1
        texto = f"D{id_dado}-{valor}"
        print(texto)
        anotaciones_70.append((x, y, ancho, alto, texto))

        cv2.putText(
            frame_etiquetado, texto, (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1
        )

cv2.imwrite(os.path.join(carpeta_figuras, "prueba_dados_etiquetados_70.png"), frame_etiquetado)

mostrar_paneles(
    [frame_etiquetado],
    ["Frame 70 con id y valor de cada dado"],
    "Prueba 7: lectura de los cinco dados",
    "07_dados_etiquetados_70.png",
)

# --- Prueba 8: verificar los centroides en el tramo candidato ---

def obtener_centros_dados(frame_reducido):
    hsv = cv2.cvtColor(frame_reducido, cv2.COLOR_BGR2HSV)
    rojo_bajo = cv2.inRange(hsv, (0, 120, 60), (10, 255, 255))
    rojo_alto = cv2.inRange(hsv, (170, 120, 60), (179, 255, 255))
    mascara = np.uint8(np.logical_or(rojo_bajo, rojo_alto)) * 255

    num_comp, _, stats_dados, centros = cv2.connectedComponentsWithStats(
        mascara, connectivity=8
    )

    centros_dados = []
    for etiqueta in range(1, num_comp):
        _, _, ancho, alto, area = stats_dados[etiqueta]
        if area >= 100 and 0.5 <= ancho / alto <= 2.0:
            centros_dados.append(centros[etiqueta])

    return np.array(centros_dados)


centros_referencia = obtener_centros_dados(frame_original_70)
tolerancia = 1.0

cap_prueba = cv2.VideoCapture("tirada_1.mp4")
if not cap_prueba.isOpened():
    raise SystemExit("No se pudo volver a abrir tirada_1.mp4")

frames_estables = []
centros_por_frame = []

for indice in range(numero_frame_total):
    ret, frame = cap_prueba.read()
    if not ret:
        break

    alto, ancho = frame.shape[:2]
    reducido = cv2.resize(frame, (ancho // 3, alto // 3))
    centros_actuales = obtener_centros_dados(reducido)
    centros_por_frame.append(centros_actuales)    

    if len(centros_referencia) != 5 or len(centros_actuales) < 5:
        print(f"Frame {indice}: {len(centros_actuales)} dados candidatos")
        continue

    indices_asociados = []
    distancias_minimas = []

    for cx, cy in centros_referencia:
        dx = centros_actuales[:, 0] - cx
        dy = centros_actuales[:, 1] - cy
        distancias = np.sqrt(dx**2 + dy**2)

        mas_cercano = int(distancias.argmin())
        indices_asociados.append(mas_cercano)
        distancias_minimas.append(distancias[mas_cercano])

    distancia_maxima = max(distancias_minimas)
    cinco_distintos = len(set(indices_asociados)) == 5
    estable = cinco_distintos and distancia_maxima <= tolerancia
        
    if estable:
        frames_estables.append(indice)

    print(f"Frame {indice}: distancia máxima {distancia_maxima:.2f} px "
          f"| Posiciones estables: {estable}")

cap_prueba.release()
print("Frames con los cinco dados estables:", frames_estables)

# --- Prueba 10: encontrar automáticamente el tramo estable ---

def coinciden_centros(referencia, actuales, tolerancia):
    if len(actuales) < 5:
        return False

    indices_asociados = []

    for cx, cy in referencia:
        dx = actuales[:, 0] - cx
        dy = actuales[:, 1] - cy
        distancias = np.sqrt(dx**2 + dy**2)

        mas_cercano = int(distancias.argmin())
        if distancias[mas_cercano] > tolerancia:
            return False

        indices_asociados.append(mas_cercano)

    return len(set(indices_asociados)) == 5


def encontrar_tramo_estable(centros_por_frame, tolerancia, minimo_consecutivos):
    mejor_inicio = None
    mejor_cantidad = 0

    for inicio, referencia in enumerate(centros_por_frame):
        if len(referencia) != 5:
            continue

        siguiente = inicio + 1
        while siguiente < len(centros_por_frame):
            actuales = centros_por_frame[siguiente]
            if not coinciden_centros(referencia, actuales, tolerancia):
                break
            siguiente += 1

        cantidad = siguiente - inicio
        if cantidad >= minimo_consecutivos and cantidad > mejor_cantidad:
            mejor_inicio = inicio
            mejor_cantidad = cantidad

    return mejor_inicio, mejor_cantidad


mejor_inicio, mejor_cantidad = encontrar_tramo_estable(
    centros_por_frame, tolerancia, minimo_consecutivos
)

if mejor_inicio is not None:
    mejor_fin = mejor_inicio + mejor_cantidad - 1
    frame_central = (mejor_inicio + mejor_fin) // 2

    print(f"Tramo automático: frames {mejor_inicio} a {mejor_fin} "
          f"({mejor_cantidad} frames)")
    print(f"Frame sugerido para leer los valores: {frame_central}")

# --- Leer los dados del frame elegido automáticamente ---

if mejor_inicio is None:
    raise SystemExit("No se encontró un tramo estable para exportar")

cap_central = cv2.VideoCapture("tirada_1.mp4")

# Leemos desde el frame 0 hasta el frame central inclusive.
for _ in range(frame_central + 1):
    ret, frame = cap_central.read()
    if not ret:
        cap_central.release()
        raise SystemExit(f"No se pudo leer el frame {frame_central}")

cap_central.release()

alto, ancho = frame.shape[:2]
frame_central_chico = cv2.resize(frame, (ancho // 3, alto // 3))

hsv = cv2.cvtColor(frame_central_chico, cv2.COLOR_BGR2HSV)
rojo_bajo = cv2.inRange(hsv, (0, 120, 60), (10, 255, 255))
rojo_alto = cv2.inRange(hsv, (170, 120, 60), (179, 255, 255))
mascara_central = np.uint8(np.logical_or(rojo_bajo, rojo_alto)) * 255

num_comp, _, stats_central, _ = cv2.connectedComponentsWithStats(
    mascara_central, connectivity=8
)

umbral_puntos_final = 175
area_minima_puntos_final = 5
anotaciones_auto = []

for etiqueta in range(1, num_comp):
    x, y, ancho, alto, area = stats_central[etiqueta]

    if area >= 100 and 0.5 <= ancho / alto <= 2.0:
        dado = frame_central_chico[y:y + alto, x:x + ancho]
        valor, _ = contar_puntos(dado, umbral_puntos_final, area_minima_puntos_final)

        id_dado = len(anotaciones_auto) + 1
        texto = f"D{id_dado}-{valor}"
        anotaciones_auto.append((x, y, ancho, alto, texto))

valores_auto = [texto for x, y, ancho, alto, texto in anotaciones_auto]
print(f"\n=== tirada_1.mp4 ===")
print(f"Frames con dados en reposo: {mejor_inicio} a {mejor_fin}")
print(f"Valores del frame {frame_central}:", ", ".join(valores_auto))

mostrar_paneles(
    [frame_central_chico, mascara_central,
     dibujar_anotaciones(frame_central_chico, anotaciones_auto)],
    [f"Frame central ({frame_central})", "Máscara roja",
     "Resultado: " + ", ".join(valores_auto)],
    "tirada_1: frame elegido automáticamente",
    "10_tirada_1_resultado.png",
)

# --- Prueba 9: exportar una tirada con los dados etiquetados ---

def exportar_tirada(ruta, inicio, fin, cuadro_central, anotaciones):
    cap_video = cv2.VideoCapture(ruta)
    if not cap_video.isOpened():
        print(f"No se pudo abrir {ruta} para exportar")
        return

    ancho_video = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_video = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_video.get(cv2.CAP_PROP_FPS)

    alto_reducido, ancho_reducido = cuadro_central.shape[:2]
    escala_x = ancho_video / ancho_reducido
    escala_y = alto_video / alto_reducido

    nombre_base = os.path.splitext(os.path.basename(ruta))[0]
    ruta_salida = os.path.join(carpeta_salida, f"{nombre_base}_etiquetada.mp4")
    salida = cv2.VideoWriter(
        ruta_salida,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (ancho_video, alto_video)
    )

    if not salida.isOpened():
        cap_video.release()
        print(f"No se pudo crear {ruta_salida}")
        return

    indice = 0

    while True:
        ret, frame = cap_video.read()
        if not ret:
            break

        if inicio <= indice <= fin:
            for x, y, ancho, alto, texto in anotaciones:
                x1 = int(x * escala_x)
                y1 = int(y * escala_y)
                x2 = int((x + ancho) * escala_x) - 1
                y2 = int((y + alto) * escala_y) - 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.putText(
                    frame, texto, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2
                )

        salida.write(frame)
        indice += 1

    cap_video.release()
    salida.release()

    print(f"Video guardado: {ruta_salida} ({indice} frames)")


exportar_tirada(
    "tirada_1.mp4", mejor_inicio, mejor_fin, frame_central_chico, anotaciones_auto
)

# --- Prueba 11: buscar tramos estables en las otras tiradas ---

for ruta in ["tirada_2.mp4", "tirada_3.mp4", "tirada_4.mp4"]:
    cap_otra = cv2.VideoCapture(ruta)
    if not cap_otra.isOpened():
        print(f"No se pudo abrir {ruta}")
        continue

    centros_otra_tirada = []

    while True:
        ret, frame = cap_otra.read()
        if not ret:
            break

        alto, ancho = frame.shape[:2]
        reducido = cv2.resize(frame, (ancho // 3, alto // 3))
        centros_otra_tirada.append(obtener_centros_dados(reducido))

    cap_otra.release()

    inicio, cantidad = encontrar_tramo_estable(
        centros_otra_tirada, tolerancia, minimo_consecutivos
    )

    if inicio is None:
        print(f"{ruta}: no se encontró un tramo estable")
        continue

    fin = inicio + cantidad - 1
    central = (inicio + fin) // 2
    print(f"\n=== {ruta} ===")
    print(f"Frames con dados en reposo: {inicio} a {fin} "
          f"({cantidad} frames), frame central {central}")

    # --- Prueba 12: comparar umbrales de puntos en el cuadro central ---
    cap_central = cv2.VideoCapture(ruta)

    for _ in range(central + 1):
        ret, frame = cap_central.read()
        if not ret:
            break

    cap_central.release()

    if not ret:
        print(f"{ruta}: no se pudo leer el frame {central}")
        continue

    alto, ancho = frame.shape[:2]
    cuadro_central = cv2.resize(frame, (ancho // 3, alto // 3))

    hsv = cv2.cvtColor(cuadro_central, cv2.COLOR_BGR2HSV)
    rojo_bajo = cv2.inRange(hsv, (0, 120, 60), (10, 255, 255))
    rojo_alto = cv2.inRange(hsv, (170, 120, 60), (179, 255, 255))
    mascara = np.uint8(np.logical_or(rojo_bajo, rojo_alto)) * 255

    num_comp, _, stats_dados, _ = cv2.connectedComponentsWithStats(
        mascara, connectivity=8
    )

    recortes_dados = []
    cajas_dados = []
    for etiqueta in range(1, num_comp):
        x, y, ancho, alto, area = stats_dados[etiqueta]
        if area >= 100 and 0.5 <= ancho / alto <= 2.0:
            recortes_dados.append(cuadro_central[y:y + alto, x:x + ancho])
            cajas_dados.append((x, y, ancho, alto))

    print(f"  Dados candidatos en el frame central: {len(recortes_dados)}")

    for umbral in [160, 175]:
        valores = []
        for dado in recortes_dados:
            valor, _ = contar_puntos(dado, umbral)
            valores.append(valor)

        print(f"  Umbral {umbral}: {valores}")

    # --- Prueba 13: comparar áreas mínimas con umbral 175 ---
    for area_minima in [3, 5, 6]:
        valores = []

        for dado in recortes_dados:
            valor, _ = contar_puntos(dado, 175, area_minima)
            valores.append(valor)

        print(f"  Área mínima {area_minima} px: {valores}")

    # --- Exportar la tirada con los parámetros elegidos ---
    anotaciones_tirada = []

    for id_dado, dado in enumerate(recortes_dados, start=1):
        x, y, ancho, alto = cajas_dados[id_dado - 1]
        valor, _ = contar_puntos(dado, umbral_puntos_final, area_minima_puntos_final)

        texto = f"D{id_dado}-{valor}"
        anotaciones_tirada.append((x, y, ancho, alto, texto))

    valores_finales = [texto for x, y, ancho, alto, texto in anotaciones_tirada]
    print(f"Valores finales de {ruta}:", ", ".join(valores_finales))

    nombre_tirada = os.path.splitext(os.path.basename(ruta))[0]
    mostrar_paneles(
        [cuadro_central, mascara,
         dibujar_anotaciones(cuadro_central, anotaciones_tirada)],
        [f"Frame central ({central})", "Máscara roja",
         "Resultado: " + ", ".join(valores_finales)],
        f"{nombre_tirada}: frame elegido automáticamente",
        f"11_{nombre_tirada}_resultado.png",
    )

    exportar_tirada(ruta, inicio, fin, cuadro_central, anotaciones_tirada)
