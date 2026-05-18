# Analisis de Poros - App Web

Esta aplicacion permite cargar una imagen microscopia (TIF, PNG, JPG, BMP), detectar poros y generar un informe visual y numerico en unidades fisicas (nm, nm2).

## Que procesamiento aplica la app

El flujo de deteccion se ejecuta por etapas. A continuacion se describe cada una:

1. Carga y normalizacion de imagen
- Se recibe la imagen cargada en la web.
- Si es TIF/TIFF se lee con `tifffile`; otros formatos se leen con OpenCV.
- Si la imagen tiene canales, se convierte a escala de grises.
- Si la profundidad no es uint8, se normaliza al rango 0-255 para estandarizar el procesamiento.

2. Seleccion de preset por tipo de muestra
- El usuario elige `100 nm` o `400 nm`.
- La app aplica un preset distinto para cada caso:
  - 100 nm: parametros orientados a poros mas grandes (menos fragmentacion).
  - 400 nm: parametros orientados a poros mas pequenos (mayor sensibilidad con control de ruido).

3. Mejora de contraste local (CLAHE)
- Se aplica CLAHE solo si el preset lo habilita (`clahe_clip > 0`).
- Objetivo: resaltar bordes y regiones oscuras de poros cuando el contraste original es bajo.

4. Suavizado gaussiano
- Se aplica `GaussianBlur` con kernel del preset.
- Objetivo: reducir ruido de alta frecuencia antes del umbralado.

5. Segmentacion binaria (deteccion inicial de poros)
- Se calculan dos mascaras candidatas:
  - Otsu invertido (`THRESH_BINARY_INV + OTSU`), porque los poros son oscuros.
  - Sauvola adaptativo (umbral local), util para iluminacion no uniforme.
- La app puntua ambas mascaras y elige automaticamente la mejor segun:
  - cantidad de objetos solidos,
  - variabilidad de areas,
  - penalizacion de formas tipo anillo o con huecos anormales.

6. Limpieza morfologica
- Se aplica cierre (`morph_close`) y apertura (`morph_open`) con kernel eliptico 3x3.
- Objetivo:
  - cierre: unir pequenos cortes internos,
  - apertura: eliminar ruido aislado.

7. Relleno de huecos pequenos
- Se aplica `remove_small_holes` con umbral `fill_hole_max_px`.
- Objetivo: rellenar huecos internos pequenos dentro del poro sin fusionar poros separados.

8. Separacion de poros tocandose (Watershed)
- Se calcula transformada de distancia euclidiana sobre la mascara.
- Se detectan maximos locales con `watershed_min_dist`.
- Se usa watershed sobre el mapa de distancia invertido para separar poros adyacentes.

9. Medicion de propiedades por poro
Para cada region etiquetada se calcula:
- id del poro,
- area,
- diametro equivalente,
- circunferencia,
- centroide,
- excentricidad.

10. Distancias interporo
- Se construye un KDTree con centroides.
- Para cada poro, se busca su vecino mas cercano (k=2, ignorando el propio punto).
- De ahi salen:
  - distancia al poro mas cercano,
  - id del poro vecino,
  - distancia media, minima y maxima reportadas en la vista.

11. Calibracion y conversion a unidades fisicas
- El usuario ingresa calibracion para:
  - 100 nm: cuantos pixeles mide la barra de 100 nm,
  - 400 nm: cuantos pixeles mide la barra de 400 nm.
- La app calcula `nm/px` segun el tamano elegido y convierte:
  - longitudes: px -> nm,
  - areas: px2 -> nm2.

12. Salidas de resultados
- Vista con imagen original y detecciones.
- Histogramas en nm (diametro y distancia interporo).
- Tabla de poros en unidades fisicas.
- Descarga de informe PDF con:
  - resumen,
  - imagenes,
  - histogramas,
  - tabla completa.

## Archivos principales
- `app.py`: app Flask, flujo de analisis y reporte PDF.
- `analisis_poros.py`: funciones de segmentacion y separacion de poros.
- `templates/index.html`: formulario de carga y calibracion.
- `templates/resultados.html`: visualizacion de resultados y descarga de PDF.

## Ejecucion
1. Instalar dependencias:
   - `pip install -r requirements-webapp.txt`
2. Ejecutar la app:
   - `python app.py`
3. Abrir en navegador:
   - `http://127.0.0.1:5000`
