"""
Análisis de Poros en Imágenes
==============================
Lee imágenes TIF de la carpeta, detecta poros, y calcula:
  - Número de poros
  - Tamaño de poro (área y diámetro equivalente)
  - Distancia interporo (entre centroides de poros vecinos)

Dependencias:
    pip install opencv-python numpy scipy matplotlib pandas scikit-image tifffile
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy import ndimage
from skimage import filters, morphology, measure, segmentation, color
from skimage.morphology import remove_small_holes
from skimage.feature import peak_local_max, blob_log
import tifffile

# ─────────────────────────────────────────────
# Configuración global
# ─────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
CARPETA_DATA = os.path.join(_BASE, "data") if os.path.isdir(os.path.join(_BASE, "data")) else _BASE
EXTENSIONES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

# Guarda imágenes de etapas intermedias (útil para ajustar parámetros)
MODO_DIAGNOSTICO = True

# ── Parámetros por grupo (nombre_subcarpeta → dict de parámetros) ──────────
# Ajusta cada grupo de forma independiente.
PARAMS_POR_GRUPO = {
    "100nm": {
        # Poros grandes (~60-100px diámetro): watershed con min_dist grande
        # para no fragmentar cada poro internamente.
        # Otsu sin CLAHE funciona bien (buen contraste natural).
        "use_watershed":     True,
        "clahe_clip":        0,
        "clahe_tile":        (8, 8),
        "blur_kernel":       (3, 3),
        "sauvola_window":    17,
        # min_dist ≈ 60-70% del radio del poro típico (~33px) → 25px
        "watershed_min_dist": 25,
        # area_min alto para no contar fragmentos de ruido
        "area_minima_px":    500,
        "area_maxima_px":    None,
        # Morfología mínima: no queremos unir poros vecinos
        "morph_close":       1,
        "morph_open":        1,
        # fill_hole pequeño: solo rellenar halo brillante interno del poro
        "fill_hole_max_px":  150,
    },
    "400nm": {
        "use_watershed":     True,
        "clahe_clip":        2.0,
        "clahe_tile":        (8, 8),
        "blur_kernel":       (5, 5),
        "sauvola_window":    25,
        "watershed_min_dist": 6,
        "area_minima_px":    20,
        "area_maxima_px":    None,
        "morph_close":       2,
        "morph_open":        1,
        "fill_hole_max_px":  60,
    },
}

# Parámetros por defecto
PARAMS_DEFAULT = {
    "use_watershed":     True,
    "clahe_clip":        2.0,
    "clahe_tile":        (8, 8),
    "blur_kernel":       (5, 5),
    "sauvola_window":    25,
    "watershed_min_dist": 5,
    "area_minima_px":    20,
    "area_maxima_px":    None,
    "morph_close":       2,
    "morph_open":        1,
    "fill_hole_max_px":  80,
}


# ─────────────────────────────────────────────
# Funciones auxiliares
# ─────────────────────────────────────────────
def cargar_imagen(ruta):
    """Carga imagen TIF (incluye 16-bit) y la devuelve como uint8 en escala de grises."""
    ext = os.path.splitext(ruta)[1].lower()
    if ext in (".tif", ".tiff"):
        img = tifffile.imread(ruta)
    else:
        img = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)

    # Convertir a 2D si tiene canales
    if img.ndim == 3:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 else img[:, :, 0]

    # Normalizar a uint8
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return img


def detectar_blobs(gris, params):
    """
    Detecta poros como blobs oscuros usando Laplacian-of-Gaussian (LoG).
    Ideal para poros bien definidos y separados.
    Devuelve (df_poros, dict_etapas).
    """
    # LoG trabaja sobre imagen normalizada [0,1]; invertimos para detectar oscuros
    img_norm = gris.astype(np.float32) / 255.0
    img_inv = 1.0 - img_norm

    blobs = blob_log(
        img_inv,
        min_sigma=params["blob_min_sigma"],
        max_sigma=params["blob_max_sigma"],
        num_sigma=params["blob_num_sigma"],
        threshold=params["blob_threshold"],
        overlap=0.4,
    )
    # blobs: columnas [y, x, sigma]; radio = sigma * sqrt(2)
    datos = []
    for i, (y, x, sigma) in enumerate(blobs, start=1):
        radio = sigma * np.sqrt(2)
        area = np.pi * radio ** 2
        datos.append({
            "id_poro": i,
            "area_px": round(area, 1),
            "diametro_equiv_px": round(2 * radio, 2),
            "centroide_y": round(y, 2),
            "centroide_x": round(x, 2),
            "excentricidad": 0.0,   # blob_log asume blobs circulares
        })

    df = pd.DataFrame(datos)

    # Imagen de diagnóstico con círculos superpuestos
    vis = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
    for _, (y, x, sigma) in enumerate(blobs):
        r = int(sigma * np.sqrt(2))
        cv2.circle(vis, (int(x), int(y)), max(r, 1), (0, 255, 0), 1)

    etapas = {
        "1_original":   gris,
        "2_inv_norm":   (img_inv * 255).astype(np.uint8),
        "3_blobs_LoG":  cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY),
    }
    return df, etapas


def segmentar_poros(gris, params):
    """
    Segmenta los poros de la imagen en escala de grises.
    Devuelve (máscara_binaria, dict_etapas) donde etapas contiene imágenes intermedias.
    """
    area_min = params["area_minima_px"]

    # 1. Ecualización CLAHE (se omite si clahe_clip == 0)
    if params["clahe_clip"] > 0:
        clahe = cv2.createCLAHE(clipLimit=params["clahe_clip"],
                                 tileGridSize=params["clahe_tile"])
        mejorada = clahe.apply(gris)
    else:
        mejorada = gris.copy()

    # 2. Suavizado gaussiano
    k = params["blur_kernel"]
    suavizada = cv2.GaussianBlur(mejorada, k, 0)

    # 3. Umbral Otsu (poros más oscuros → invertido)
    _, mascara_otsu = cv2.threshold(suavizada, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Umbral local Sauvola
    umbral_sauvola = filters.threshold_sauvola(suavizada,
                                               window_size=params["sauvola_window"])
    mascara_sauvola = (suavizada < umbral_sauvola).astype(np.uint8) * 255

    # 5. Seleccionar la mejor máscara:
    #    - favorece muchos objetos de tamaño similar
    #    - penaliza máscaras con muchos objetos tipo anillo (Euler < 0 o con agujeros)
    def puntaje(mask):
        mask_bin = ndimage.binary_fill_holes(mask > 0)
        etiquetas, n = ndimage.label(mask_bin)
        if n == 0:
            return 0
        props = measure.regionprops(etiquetas)
        areas = [p.area for p in props if p.area >= area_min]
        if not areas:
            return 0
        # Número de objetos con solidez alta (no anillos): solidez = area/convex_area
        n_solidos = sum(1 for p in props
                        if p.area >= area_min and p.solidity > 0.65)
        return n_solidos / (1 + np.std(areas) / (np.mean(areas) + 1e-6))

    usar_otsu = puntaje(mascara_otsu) >= puntaje(mascara_sauvola)
    mascara = mascara_otsu if usar_otsu else mascara_sauvola
    metodo_usado = "Otsu" if usar_otsu else "Sauvola"

    # 6. Morfología
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel,
                               iterations=params["morph_close"])
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel,
                               iterations=params["morph_open"])

    # 7. Rellenar sólo huecos pequeños (interior de poros/donuts).
    #    remove_small_holes rellena agujeros de área <= fill_hole_max_px
    #    sin tocar el fondo entre poros (los gaps entre poros son grandes).
    fill_max = params["fill_hole_max_px"]
    mascara_bin = mascara > 0
    mascara_filled = remove_small_holes(mascara_bin,
                                        max_size=fill_max).astype(np.uint8) * 255

    etapas = {
        "1_original":               gris,
        "2_clahe":                  mejorada,
        "3_blur":                   suavizada,
        f"4_mascara_{metodo_usado}": mascara,
        "5_fill_holes":             mascara_filled,
    }
    return mascara_filled > 0, etapas


def separar_poros_watershed(mascara, params):
    """Aplica watershed para separar poros que se tocan."""
    dist = ndimage.distance_transform_edt(mascara)

    coordenadas = peak_local_max(dist,
                                 min_distance=params["watershed_min_dist"],
                                 labels=mascara)
    marcadores = np.zeros(dist.shape, dtype=bool)
    marcadores[tuple(coordenadas.T)] = True
    marcadores, _ = ndimage.label(marcadores)

    etiquetas = segmentation.watershed(-dist, marcadores, mask=mascara)
    return etiquetas


def analizar_poros(etiquetas, params):
    """Extrae propiedades de cada poro etiquetado."""
    area_min = params["area_minima_px"]
    area_max = params["area_maxima_px"]
    props = measure.regionprops(etiquetas)
    datos = []
    for p in props:
        if p.area < area_min:
            continue
        if area_max is not None and p.area > area_max:
            continue
        diametro = 2 * np.sqrt(p.area / np.pi)
        datos.append({
            "id_poro": p.label,
            "area_px": p.area,
            "diametro_equiv_px": round(diametro, 2),
            "centroide_y": round(p.centroid[0], 2),
            "centroide_x": round(p.centroid[1], 2),
            "excentricidad": round(p.eccentricity, 4),
        })
    return pd.DataFrame(datos)


def calcular_distancias_interporo(df_poros):
    """Calcula la distancia al poro más cercano (centro a centro) para cada poro.
    Agrega columnas: distancia_poro_cercano_px  e  id_poro_cercano.
    """
    if len(df_poros) < 2:
        df_poros["distancia_poro_cercano_px"] = np.nan
        df_poros["id_poro_cercano"]            = np.nan
        return df_poros

    coords = df_poros[["centroide_y", "centroide_x"]].values
    ids    = df_poros["id_poro"].values
    arbol  = KDTree(coords)
    # k=2: índice 0 = el propio poro (dist=0), índice 1 = el más cercano
    distancias, indices = arbol.query(coords, k=2)
    df_poros = df_poros.copy()
    df_poros["distancia_poro_cercano_px"] = np.round(distancias[:, 1], 2)
    df_poros["id_poro_cercano"]            = ids[indices[:, 1]]
    return df_poros


def guardar_diagnostico(etapas, nombre_base, carpeta_salida):
    """Guarda las etapas intermedias de segmentación en un único panel."""
    n = len(etapas)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (titulo, img) in zip(axes, etapas.items()):
        ax.imshow(img, cmap="gray")
        ax.set_title(titulo, fontsize=7)
        ax.axis("off")
    plt.suptitle(f"Diagnóstico: {nombre_base}", fontsize=9, fontweight="bold")
    plt.tight_layout()
    ruta = os.path.join(carpeta_salida, f"{nombre_base}_diagnostico.png")
    plt.savefig(ruta, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Diagnóstico: {ruta}")


def guardar_visualizacion(gris, etiquetas, df_poros, nombre_base, carpeta_salida):
    """Genera y guarda imagen anotada con los poros detectados."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(nombre_base, fontsize=13, fontweight="bold")

    # Tamaño de fuente adaptativo según cantidad de poros
    n = len(df_poros)
    fontsize_id = max(3, min(8, 180 // max(n, 1)))

    # Panel 1: Imagen original
    axes[0].imshow(gris, cmap="gray")
    axes[0].set_title("Imagen Original")
    axes[0].axis("off")

    # Panel 2: Máscara coloreada + ID de cada poro
    if etiquetas is not None:
        overlay = color.label2rgb(etiquetas, image=gris, bg_label=0, alpha=0.4)
    else:
        overlay = cv2.cvtColor(gris, cv2.COLOR_GRAY2RGB) / 255.0
    axes[1].imshow(overlay)
    axes[1].set_title(f"Poros detectados: {n}")
    axes[1].axis("off")
    if not df_poros.empty:
        for _, fila in df_poros.iterrows():
            axes[1].text(
                fila["centroide_x"], fila["centroide_y"],
                str(int(fila["id_poro"])),
                fontsize=fontsize_id, color="white", fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.45, lw=0),
            )

    # Panel 3: Líneas al vecino más cercano + ID en centroide
    axes[2].imshow(gris, cmap="gray")
    axes[2].set_title("Distancias al poro más cercano")
    axes[2].axis("off")
    if not df_poros.empty:
        # Diccionario rápido: id_poro → (cx, cy)
        pos = {int(r["id_poro"]): (r["centroide_x"], r["centroide_y"])
               for _, r in df_poros.iterrows()}

        has_vecino = "id_poro_cercano" in df_poros.columns

        for _, fila in df_poros.iterrows():
            cx, cy = fila["centroide_x"], fila["centroide_y"]

            # Línea al vecino más cercano
            if has_vecino and not pd.isna(fila.get("id_poro_cercano")):
                vid = int(fila["id_poro_cercano"])
                if vid in pos:
                    vx, vy = pos[vid]
                    axes[2].plot([cx, vx], [cy, vy],
                                 color="cyan", linewidth=0.6, alpha=0.7, zorder=3)
                    # Etiqueta de distancia en el punto medio
                    mx, my = (cx + vx) / 2, (cy + vy) / 2
                    axes[2].text(
                        mx, my,
                        f"{fila['distancia_poro_cercano_px']:.0f}px",
                        fontsize=max(2, fontsize_id - 1), color="cyan",
                        ha="center", va="center", alpha=0.85,
                    )

            # Punto + ID del poro
            axes[2].plot(cx, cy, "r.", markersize=3, zorder=5)
            axes[2].text(
                cx, cy - fontsize_id * 0.8,
                str(int(fila["id_poro"])),
                fontsize=fontsize_id, color="yellow", fontweight="bold",
                ha="center", va="bottom", zorder=6,
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.4, lw=0),
            )

    plt.tight_layout()
    ruta_fig = os.path.join(carpeta_salida, f"{nombre_base}_resultado.png")
    plt.savefig(ruta_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Visualización guardada: {ruta_fig}")


def guardar_grafico_comparativo(df_total, carpeta_salida):
    """Gráfico de barras comparando diámetro e interporo entre grupos."""
    grupos = df_total["grupo"].unique()
    x = np.arange(len(grupos))
    ancho = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Comparativa entre grupos", fontsize=13, fontweight="bold")

    # Agrupar por grupo (promedio de las imágenes del grupo)
    agrup = df_total.groupby("grupo").agg(
        diam_mean=("diametro_medio_px", "mean"),
        diam_std=("diametro_medio_px", "std"),
        dist_mean=("dist_interporo_media_px", "mean"),
        dist_std=("dist_interporo_media_px", "std"),
        n_poros=("n_poros", "sum"),
    ).reindex(sorted(grupos))

    # Diámetro
    axes[0].bar(range(len(agrup)), agrup["diam_mean"], yerr=agrup["diam_std"].fillna(0),
                capsize=5, color=["steelblue", "tomato"][:len(agrup)])
    axes[0].set_xticks(range(len(agrup)))
    axes[0].set_xticklabels(agrup.index)
    axes[0].set_ylabel("Diámetro equiv. medio (px)")
    axes[0].set_title("Tamaño de poro")

    # Distancia interporo
    axes[1].bar(range(len(agrup)), agrup["dist_mean"], yerr=agrup["dist_std"].fillna(0),
                capsize=5, color=["steelblue", "tomato"][:len(agrup)])
    axes[1].set_xticks(range(len(agrup)))
    axes[1].set_xticklabels(agrup.index)
    axes[1].set_ylabel("Distancia interporo media (px)")
    axes[1].set_title("Distancia interporo")

    plt.tight_layout()
    ruta = os.path.join(carpeta_salida, "comparativa_grupos.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico comparativo: {ruta}")


def imprimir_resumen(nombre, df_poros):
    n = len(df_poros)
    if n == 0:
        print("  No se detectaron poros.")
        return
    print(f"  Poros detectados          : {n}")
    print(f"  Área media (px²)          : {df_poros['area_px'].mean():.1f} ± {df_poros['area_px'].std():.1f}")
    print(f"  Diámetro equiv. medio (px): {df_poros['diametro_equiv_px'].mean():.1f} ± {df_poros['diametro_equiv_px'].std():.1f}")
    if "distancia_poro_cercano_px" in df_poros:
        d = df_poros["distancia_poro_cercano_px"].dropna()
        print(f"  Dist. interporo media (px): {d.mean():.1f} ± {d.std():.1f}")


# ─────────────────────────────────────────────
# Programa principal
# ─────────────────────────────────────────────
def procesar_carpeta(carpeta, nombre_grupo):
    """Procesa todas las imágenes de una carpeta y devuelve el resumen del grupo."""
    carpeta_salida = os.path.join(carpeta, "resultados")
    os.makedirs(carpeta_salida, exist_ok=True)

    # Seleccionar parámetros para este grupo
    params = PARAMS_POR_GRUPO.get(nombre_grupo, PARAMS_DEFAULT)
    ws = "watershed" if params.get("use_watershed", True) else "componentes_conectados"
    print(f"  Parámetros: area_min={params['area_minima_px']}px | "
          f"area_max={params['area_maxima_px']}px | "
          f"fill_hole_max={params['fill_hole_max_px']}px | "
          f"segmentacion={ws} | "
          f"clahe_clip={params['clahe_clip']} | "
          f"blur={params['blur_kernel']}")

    imagenes = [
        f for f in os.listdir(carpeta)
        if os.path.splitext(f)[1].lower() in EXTENSIONES
        and not f.startswith(".")
    ]

    if not imagenes:
        print(f"  Sin imágenes en {carpeta}")
        return []

    resumen_grupo = []

    for nombre_archivo in sorted(imagenes):
        ruta = os.path.join(carpeta, nombre_archivo)
        nombre_base = os.path.splitext(nombre_archivo)[0]
        print(f"\n  {'─'*54}")
        print(f"  Imagen: {nombre_archivo}")
        print(f"  {'─'*54}")

        try:
            gris = cargar_imagen(ruta)
            print(f"  Resolución: {gris.shape[1]} x {gris.shape[0]} px")

            mascara, etapas = segmentar_poros(gris, params)

            if params.get("use_watershed", True):
                etiquetas = separar_poros_watershed(mascara, params)
            else:
                # Componentes conectados directamente: no fragmenta poros separados
                etiquetas, n = ndimage.label(mascara)
                print(f"  [componentes conectados, sin watershed]")

            df_poros = analizar_poros(etiquetas, params)
            df_poros = calcular_distancias_interporo(df_poros)

            imprimir_resumen(nombre_base, df_poros)

            ruta_csv = os.path.join(carpeta_salida, f"{nombre_base}_poros.csv")
            df_poros.to_csv(ruta_csv, index=False)
            print(f"  Datos: {ruta_csv}")

            if MODO_DIAGNOSTICO:
                guardar_diagnostico(etapas, nombre_base, carpeta_salida)

            guardar_visualizacion(gris, etiquetas, df_poros, nombre_base, carpeta_salida)

            if not df_poros.empty:
                resumen_grupo.append({
                    "grupo": nombre_grupo,
                    "imagen": nombre_archivo,
                    "n_poros": len(df_poros),
                    "area_media_px": round(df_poros["area_px"].mean(), 2),
                    "area_std_px": round(df_poros["area_px"].std(), 2),
                    "diametro_medio_px": round(df_poros["diametro_equiv_px"].mean(), 2),
                    "diametro_std_px": round(df_poros["diametro_equiv_px"].std(), 2),
                    "dist_interporo_media_px": round(df_poros["distancia_poro_cercano_px"].mean(), 2),
                    "dist_interporo_std_px": round(df_poros["distancia_poro_cercano_px"].std(), 2),
                })

        except Exception as e:
            print(f"  ERROR: {e}")

    # Resumen por grupo
    if resumen_grupo:
        df_grupo = pd.DataFrame(resumen_grupo)
        ruta_res = os.path.join(carpeta_salida, f"resumen_{nombre_grupo}.csv")
        df_grupo.to_csv(ruta_res, index=False)
        print(f"\n  Resumen del grupo guardado: {ruta_res}")

    return resumen_grupo


def main():
    # Detectar subcarpetas con imágenes dentro de data/
    subcarpetas = sorted([
        d for d in os.listdir(CARPETA_DATA)
        if os.path.isdir(os.path.join(CARPETA_DATA, d)) and d != "resultados"
    ])

    # Si no hay subcarpetas, usar data/ directamente como un único grupo
    if not subcarpetas:
        subcarpetas_rutas = [(CARPETA_DATA, "general")]
    else:
        subcarpetas_rutas = [
            (os.path.join(CARPETA_DATA, d), d) for d in subcarpetas
        ]

    resumen_total = []

    for carpeta, nombre_grupo in subcarpetas_rutas:
        print(f"\n{'='*60}")
        print(f"GRUPO: {nombre_grupo.upper()}")
        print(f"{'='*60}")
        resumen_total.extend(procesar_carpeta(carpeta, nombre_grupo))

    # Resumen global comparando todos los grupos
    if resumen_total:
        df_total = pd.DataFrame(resumen_total)
        carpeta_salida_global = os.path.join(CARPETA_DATA, "resultados")
        os.makedirs(carpeta_salida_global, exist_ok=True)
        ruta_global = os.path.join(carpeta_salida_global, "resumen_global.csv")
        df_total.to_csv(ruta_global, index=False)

        print(f"\n{'='*60}")
        print("RESUMEN COMPARATIVO ENTRE GRUPOS")
        print(f"{'='*60}")
        print(df_total.to_string(index=False))
        print(f"\nResumen global: {ruta_global}")

        # Gráfico comparativo de diámetro e interporo por grupo
        guardar_grafico_comparativo(df_total, carpeta_salida_global)


if __name__ == "__main__":
    main()
