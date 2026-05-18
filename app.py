import base64
import io
import os
import uuid

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import tifffile
from flask import Flask, render_template, request, send_file
from scipy import ndimage
from scipy.spatial import KDTree
from skimage import color, measure

from analisis_poros import segmentar_poros, separar_poros_watershed


app = Flask(__name__)
REPORT_CACHE = {}


CALIBRATION_DEFAULTS = {
    "pixels_100nm": 100.0,
    "pixels_400nm": 77.0,
}

NOMINAL_BAR_NM = {
    "100nm": 100.0,
    "400nm": 400.0,
}


# Parametros ajustados para uso segun tamaño de imagen.
WEBAPP_PRESETS = {
    "100nm": {
        "use_watershed": True,
        "clahe_clip": 0.0,
        "clahe_tile": (8, 8),
        "blur_kernel": (3, 3),
        "sauvola_window": 17,
        "watershed_min_dist": 28,
        "area_minima_px": 650,
        "area_maxima_px": None,
        "morph_close": 1,
        "morph_open": 1,
        "fill_hole_max_px": 140,
    },
    "400nm": {
        "use_watershed": True,
        "clahe_clip": 1.8,
        "clahe_tile": (8, 8),
        "blur_kernel": (5, 5),
        "sauvola_window": 25,
        "watershed_min_dist": 8,
        "area_minima_px": 35,
        "area_maxima_px": None,
        "morph_close": 2,
        "morph_open": 2,
        "fill_hole_max_px": 45,
    },
}


def cargar_imagen_subida(file_storage):
    """Carga imagen desde upload y la devuelve como uint8 en grises."""
    data = file_storage.read()
    if not data:
        raise ValueError("No se recibio contenido de imagen.")

    ext = os.path.splitext(file_storage.filename or "")[1].lower()
    if ext in (".tif", ".tiff"):
        img = tifffile.imread(io.BytesIO(data))
    else:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError("No se pudo leer la imagen subida.")

    if img.ndim == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            img = img[:, :, 0]

    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return img


def analizar_etiquetas(etiquetas, area_min, area_max):
    """Extrae tabla de poros y distancias al vecino mas cercano."""
    props = measure.regionprops(etiquetas)
    filas = []

    for p in props:
        if p.area < area_min:
            continue
        if area_max is not None and p.area > area_max:
            continue

        diametro = 2 * np.sqrt(p.area / np.pi)
        filas.append({
            "id_poro": int(p.label),
            "area_px": float(p.area),
            "diametro_equiv_px": float(diametro),
            "circunferencia_px": float(p.perimeter),
            "centroide_y": float(p.centroid[0]),
            "centroide_x": float(p.centroid[1]),
            "excentricidad": float(p.eccentricity),
        })

    df = pd.DataFrame(filas)
    if df.empty:
        return df

    if len(df) < 2:
        df["distancia_poro_cercano_px"] = np.nan
        df["id_poro_cercano"] = np.nan
        return df

    coords = df[["centroide_y", "centroide_x"]].to_numpy()
    ids = df["id_poro"].to_numpy()
    tree = KDTree(coords)
    dists, idx = tree.query(coords, k=2)

    df["distancia_poro_cercano_px"] = dists[:, 1]
    df["id_poro_cercano"] = ids[idx[:, 1]]
    return df


def image_to_base64(img):
    """Convierte arreglo de imagen en PNG base64 para HTML."""
    ok, buffer = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("No se pudo convertir imagen a PNG.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def generar_overlay_base64(gris, etiquetas, df):
    """Crea imagen con detecciones y anotaciones."""
    overlay = color.label2rgb(etiquetas, image=gris, bg_label=0, alpha=0.35)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(overlay)
    ax.set_title("Detecciones de poros")
    ax.axis("off")

    posiciones = {}
    if not df.empty:
        for _, row in df.iterrows():
            posiciones[int(row["id_poro"])] = (row["centroide_x"], row["centroide_y"])

        for _, row in df.iterrows():
            cx = row["centroide_x"]
            cy = row["centroide_y"]
            pid = int(row["id_poro"])
            ax.text(
                cx,
                cy,
                str(pid),
                color="white",
                fontsize=7,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.1", "fc": "black", "alpha": 0.45, "lw": 0},
            )

            vecino = row.get("id_poro_cercano")
            if pd.notna(vecino):
                vid = int(vecino)
                if vid in posiciones:
                    vx, vy = posiciones[vid]
                    ax.plot([cx, vx], [cy, vy], color="cyan", linewidth=0.5, alpha=0.6)

    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def generar_histograma_base64(df, nm_per_px):
    """Genera histogramas de diametro y distancia en nm."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    diam_nm = df["diametro_equiv_px"] * nm_per_px
    axes[0].hist(diam_nm, bins=20, color="#3b82f6", edgecolor="black", alpha=0.8)
    axes[0].set_title("Histograma de diametro equivalente")
    axes[0].set_xlabel("Diametro (nm)")
    axes[0].set_ylabel("Frecuencia")

    dist = (df["distancia_poro_cercano_px"] * nm_per_px).dropna()
    if len(dist) > 0:
        axes[1].hist(dist, bins=20, color="#10b981", edgecolor="black", alpha=0.8)
    axes[1].set_title("Histograma de distancia interporo")
    axes[1].set_xlabel("Distancia (nm)")
    axes[1].set_ylabel("Frecuencia")

    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def parse_params(form):
    """Selecciona preset segun tamano de poro objetivo."""
    tamano = form.get("tamano_nm", "100nm")
    if tamano not in WEBAPP_PRESETS:
        tamano = "100nm"
    return tamano, WEBAPP_PRESETS[tamano].copy()


def parse_calibration(form):
    """Lee calibracion ingresada en pixeles para 100nm y 400nm."""
    px_100 = float(form.get("pixels_100nm", CALIBRATION_DEFAULTS["pixels_100nm"]))
    px_400 = float(form.get("pixels_400nm", CALIBRATION_DEFAULTS["pixels_400nm"]))
    if px_100 <= 0 or px_400 <= 0:
        raise ValueError("Los pixeles de calibracion deben ser mayores a 0.")
    return {
        "pixels_100nm": px_100,
        "pixels_400nm": px_400,
    }


def nm_per_px_for_size(tamano_nm, calibration):
    """Calcula factor de conversion nm/px para el tamano seleccionado."""
    if tamano_nm == "100nm":
        return NOMINAL_BAR_NM["100nm"] / calibration["pixels_100nm"]
    return NOMINAL_BAR_NM["400nm"] / calibration["pixels_400nm"]


def tabla_con_unidades(df, nm_per_px):
    """Arma tabla final solo con unidades fisicas (nm)."""
    if df.empty:
        return df

    df_v = pd.DataFrame({
        "id_poro": df["id_poro"],
        "area_nm2": df["area_px"] * (nm_per_px ** 2),
        "diametro_equiv_nm": df["diametro_equiv_px"] * nm_per_px,
        "circunferencia_nm": df["circunferencia_px"] * nm_per_px,
        "distancia_poro_cercano_nm": df["distancia_poro_cercano_px"] * nm_per_px,
        "id_poro_cercano": df["id_poro_cercano"],
        "excentricidad": df["excentricidad"],
    })
    return df_v


def _fmt_nm(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f} nm"


def generar_pdf_bytes(original_b64, overlay_b64, hist_b64, metricas, tamano_nm, calibration, nm_per_px, df_view):
    """Genera el informe PDF en memoria."""
    pdf_buffer = io.BytesIO()

    with PdfPages(pdf_buffer) as pdf:
        # Pagina 1: resumen e imagenes principales
        fig = plt.figure(figsize=(11.69, 8.27))
        grid = fig.add_gridspec(2, 2, height_ratios=[0.9, 2.1])

        ax_txt = fig.add_subplot(grid[0, :])
        ax_txt.axis("off")
        resumen = [
            "Informe de Analisis de Poros",
            f"Preset aplicado: {tamano_nm}",
            (
                "Calibracion: "
                f"100 nm = {calibration['pixels_100nm']} px | "
                f"400 nm = {calibration['pixels_400nm']} px"
            ),
            f"Factor usado: {nm_per_px:.4f} nm/px",
            f"Poros detectados: {metricas['n_poros']}",
            f"Circunferencia media: {_fmt_nm(metricas['circ_media_nm'])}",
            f"Distancia media: {_fmt_nm(metricas['dist_media_nm'])}",
            f"Distancia minima: {_fmt_nm(metricas['dist_min_nm'])}",
            f"Distancia maxima: {_fmt_nm(metricas['dist_max_nm'])}",
        ]
        ax_txt.text(0.01, 0.95, "\n".join(resumen), va="top", fontsize=10)

        img_original = plt.imread(io.BytesIO(base64.b64decode(original_b64)), format="png")
        img_overlay = plt.imread(io.BytesIO(base64.b64decode(overlay_b64)), format="png")

        ax_o = fig.add_subplot(grid[1, 0])
        ax_o.imshow(img_original)
        ax_o.set_title("Imagen original")
        ax_o.axis("off")

        ax_d = fig.add_subplot(grid[1, 1])
        ax_d.imshow(img_overlay)
        ax_d.set_title("Imagen con detecciones")
        ax_d.axis("off")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Pagina 2: histogramas
        if hist_b64:
            fig_h, ax_h = plt.subplots(figsize=(11.69, 8.27))
            ax_h.imshow(plt.imread(io.BytesIO(base64.b64decode(hist_b64)), format="png"))
            ax_h.set_title("Histogramas")
            ax_h.axis("off")
            fig_h.tight_layout()
            pdf.savefig(fig_h)
            plt.close(fig_h)

        # Paginas de tabla completas (sin truncar filas)
        if not df_view.empty:
            chunk_size = 28
            total_rows = len(df_view)

            for start in range(0, total_rows, chunk_size):
                end = min(start + chunk_size, total_rows)
                chunk = df_view.iloc[start:end].round(3)

                fig_t, ax_t = plt.subplots(figsize=(11.69, 8.27))
                ax_t.axis("off")
                table = ax_t.table(
                    cellText=chunk.values,
                    colLabels=chunk.columns,
                    loc="center",
                    cellLoc="center",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(7)
                table.scale(1, 1.25)

                titulo = f"Tabla de poros ({start + 1}-{end} de {total_rows})"
                ax_t.set_title(titulo, fontsize=11, pad=12)

                fig_t.tight_layout()
                pdf.savefig(fig_t)
                plt.close(fig_t)

    pdf_buffer.seek(0)
    return pdf_buffer.read()


@app.route("/descargar_pdf/<token>", methods=["GET"])
def descargar_pdf(token):
    reporte = REPORT_CACHE.get(token)
    if reporte is None:
        return "El informe ya no esta disponible. Genera resultados nuevamente.", 404

    return send_file(
        io.BytesIO(reporte["bytes"]),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=reporte["filename"],
    )


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", calibration_defaults=CALIBRATION_DEFAULTS)


@app.route("/resultados", methods=["POST"])
def resultados():
    archivo = request.files.get("imagen")
    if archivo is None or archivo.filename == "":
        return render_template("index.html", error="Debes subir una imagen primero.", calibration_defaults=CALIBRATION_DEFAULTS)

    tamano_nm, params = parse_params(request.form)

    try:
        calibration = parse_calibration(request.form)
        nm_per_px = nm_per_px_for_size(tamano_nm, calibration)
        gris = cargar_imagen_subida(archivo)
        mascara, _ = segmentar_poros(gris, params)

        if params.get("use_watershed", True):
            etiquetas = separar_poros_watershed(mascara, params)
        else:
            etiquetas, _ = ndimage.label(mascara)

        df = analizar_etiquetas(etiquetas, params["area_minima_px"], params["area_maxima_px"])

        original_b64 = image_to_base64(gris)
        overlay_b64 = generar_overlay_base64(gris, etiquetas, df)

        hist_b64 = None
        tabla_html = "<p>No se detectaron poros para los parametros elegidos.</p>"
        metricas = {
            "n_poros": 0,
            "circ_media_nm": np.nan,
            "dist_media_nm": np.nan,
            "dist_min_nm": np.nan,
            "dist_max_nm": np.nan,
        }

        if not df.empty:
            hist_b64 = generar_histograma_base64(df, nm_per_px)
            df_view = tabla_con_unidades(df, nm_per_px)
            tabla_html = df_view.round(3).to_html(index=False, classes="tabla-datos", border=0)

            d = df["distancia_poro_cercano_px"].dropna()
            metricas = {
                "n_poros": int(len(df)),
                "circ_media_nm": float(df["circunferencia_px"].mean() * nm_per_px),
                "dist_media_nm": float(d.mean() * nm_per_px) if len(d) > 0 else np.nan,
                "dist_min_nm": float(d.min() * nm_per_px) if len(d) > 0 else np.nan,
                "dist_max_nm": float(d.max() * nm_per_px) if len(d) > 0 else np.nan,
            }

        pdf_bytes = generar_pdf_bytes(
            original_b64,
            overlay_b64,
            hist_b64,
            metricas,
            tamano_nm,
            calibration,
            nm_per_px,
            df_view if not df.empty else pd.DataFrame(),
        )
        pdf_token = str(uuid.uuid4())
        REPORT_CACHE[pdf_token] = {
            "bytes": pdf_bytes,
            "filename": f"informe_poros_{tamano_nm}.pdf",
        }
        if len(REPORT_CACHE) > 20:
            REPORT_CACHE.pop(next(iter(REPORT_CACHE)))

        return render_template(
            "resultados.html",
            original_b64=original_b64,
            overlay_b64=overlay_b64,
            hist_b64=hist_b64,
            tabla_html=tabla_html,
            metricas=metricas,
            tamano_nm=tamano_nm,
            calibration=calibration,
            nm_per_px=nm_per_px,
            pdf_token=pdf_token,
        )
    except Exception as ex:
        return render_template("index.html", error=f"Error procesando imagen: {ex}", calibration_defaults=CALIBRATION_DEFAULTS)


if __name__ == "__main__":
    app.run(debug=True)
