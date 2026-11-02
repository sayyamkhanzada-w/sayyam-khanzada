"""
Fabric Cover Factor Analyzer
Streamlit application for educational estimation of woven-fabric cover factor.

Method overview:
1. Load the uploaded image and apply EXIF orientation.
2. Convert RGB image to grayscale.
3. Enhance contrast and calculate directional Sobel gradients.
4. Build one-dimensional edge-energy profiles and estimate their periodicity
   using autocorrelation.
5. Convert estimated pixel spacing to EPI/PPI using user-supplied pixels/inch.
6. Calculate cover factors using the requested Peirce equations.

Important: a photo has no inherent physical inch scale. Calibration is required.
The detected lines are spacing guides, not guaranteed yarn-centre detections.
"""
import io
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image, ImageOps


st.set_page_config(
    page_title="Fabric Cover Factor Analyzer",
    page_icon="🧵",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {font-size:2.25rem;font-weight:800;margin-bottom:0.15rem;}
    .sub-title {color:#64748b;margin-bottom:1.4rem;}
    .member-box {
        padding: 16px 18px; border-radius: 12px;
        background: linear-gradient(120deg,#eff6ff,#f0fdfa);
        border: 1px solid #cbdff1; margin-top: 12px;
    }
    .member-box h3 {margin:0 0 10px 0;color:#17324d;}
    .member-row {font-size:1.05rem;padding:4px 0;color:#243b53;}
    .small-note {font-size:0.88rem;color:#64748b;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🧵 Fabric Cover Factor Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Estimate EPI, PPI and Peirce cover factor from a close-up woven-fabric image.</div>',
    unsafe_allow_html=True,
)


def estimate_period(profile: np.ndarray):
    """Estimate dominant repeat spacing (pixels) from a 1-D edge profile."""
    signal = np.asarray(profile, dtype=np.float32)
    if signal.size < 24:
        return None

    # Remove gradual illumination changes and normalize.
    smooth = cv2.GaussianBlur(signal.reshape(1, -1), (0, 0), 5).ravel()
    signal = signal - smooth
    signal -= signal.mean()
    standard_deviation = signal.std()
    if standard_deviation < 1e-5:
        return None
    signal /= standard_deviation

    # Hanning window reduces edge artefacts in autocorrelation.
    signal *= np.hanning(signal.size).astype(np.float32)
    correlation = np.correlate(signal, signal, mode="full")[signal.size - 1:]
    if correlation[0] <= 0:
        return None
    correlation /= correlation[0]

    min_lag = 3
    max_lag = min(signal.size // 2, 250)
    if max_lag <= min_lag:
        return None

    segment = correlation[min_lag:max_lag]
    peaks = [
        (float(segment[i]), i + min_lag)
        for i in range(1, len(segment) - 1)
        if segment[i] > segment[i - 1] and segment[i] >= segment[i + 1]
    ]
    if not peaks:
        return None

    peak_strength, lag = max(peaks, key=lambda item: item[0])
    if peak_strength < 0.08:
        return None
    return float(lag)


def process_fabric_image(pil_image: Image.Image):
    """
    Return grayscale image, annotated image and estimated directional spacing.

    Assumption: warp yarns are oriented vertically in the uploaded image and
    weft yarns horizontally. Rotate the image before upload if necessary.
    """
    rgb = np.array(ImageOps.exif_transpose(pil_image).convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE improves local contrast; blur reduces sensor noise.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # X gradient responds to vertical boundaries; Y gradient to horizontal.
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)

    vertical_profile = np.mean(np.abs(grad_x), axis=0)
    horizontal_profile = np.mean(np.abs(grad_y), axis=1)

    vertical_spacing = estimate_period(vertical_profile)
    horizontal_spacing = estimate_period(horizontal_profile)

    annotated = bgr.copy()
    height, width = gray.shape

    # Draw repeated guide lines to visualize the estimated spacing.
    if vertical_spacing:
        step = max(1, int(round(vertical_spacing)))
        for x in range(step, width, step):
            cv2.line(annotated, (x, 0), (x, height - 1), (0, 220, 255), 1)
    if horizontal_spacing:
        step = max(1, int(round(horizontal_spacing)))
        for y in range(step, height, step):
            cv2.line(annotated, (0, y), (width - 1, y), (255, 90, 0), 1)

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return gray, annotated_rgb, vertical_spacing, horizontal_spacing


def calculate_cover(epi, ppi, warp_count, weft_count):
    """Calculate requested Peirce cover factors and classification."""
    warp_cover = epi / math.sqrt(warp_count)
    weft_cover = ppi / math.sqrt(weft_count)
    total_cover = warp_cover + weft_cover - (warp_cover * weft_cover) / 28.0
    fractional_cover = total_cover / 28.0
    percent_cover = fractional_cover * 100.0

    if total_cover < 15:
        classification = "Loose"
    elif total_cover <= 25:
        classification = "Medium"
    else:
        classification = "Tight"

    return {
        "warp_cover": warp_cover,
        "weft_cover": weft_cover,
        "total_cover": total_cover,
        "fractional_cover": fractional_cover,
        "percent_cover": percent_cover,
        "classification": classification,
    }


# Sidebar settings and group attribution.
with st.sidebar:
    st.header("⚙️ Test settings")
    warp_count = st.number_input(
        "Warp Count (Ne)", min_value=0.01, value=40.0, step=1.0
    )
    weft_count = st.number_input(
        "Weft Count (Ne)", min_value=0.01, value=40.0, step=1.0
    )
    calibration = st.slider(
        "Calibration (pixels per inch)",
        min_value=20,
        max_value=2000,
        value=300,
        step=10,
        help="Use a known 1-inch reference measured in the same plane as the fabric.",
    )
    st.caption("Calibration is required to convert pixel spacing into threads per inch.")

    st.markdown(
        """
        <div class="member-box">
          <h3>🧶 Group Members / Project By</h3>
          <div class="member-row">👤 <b>24-23TE055</b></div>
          <div class="member-row">👤 <b>24TE103</b></div>
          <div class="member-row">👤 <b>24TE052</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

uploaded_file = st.file_uploader(
    "📤 Upload a close-up woven fabric image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    help="For best results, use a sharp, straight-on image with warp vertical and weft horizontal.",
)

if uploaded_file is None:
    st.info("Upload a fabric image to begin the analysis.")
    st.markdown(
        """
        **Recommended photography:** keep the camera parallel to the fabric,
        use even diffuse lighting, avoid folds and shadows, and include a
        known scale in the same plane for calibration.
        """
    )
else:
    try:
        original = Image.open(uploaded_file)
        original = ImageOps.exif_transpose(original).convert("RGB")

        left, right = st.columns(2)
        with left:
            st.subheader("Uploaded fabric image")
            st.image(original, use_container_width=True)

        if st.button("🔬 Analyze fabric image", type="primary", use_container_width=True):
            if warp_count <= 0 or weft_count <= 0 or calibration <= 0:
                st.error("Yarn counts and calibration must be greater than zero.")
            elif original.width < 40 or original.height < 40:
                st.error("Image is too small. Upload a larger close-up image.")
            else:
                with st.spinner("Processing image and estimating thread spacing..."):
                    gray, annotated, warp_spacing_px, weft_spacing_px = process_fabric_image(original)

                if not warp_spacing_px or not weft_spacing_px:
                    st.error(
                        "Could not reliably estimate both thread directions. "
                        "Try a sharper, evenly lit, straight-on close-up image."
                    )
                else:
                    # Warp runs vertically: spacing measured across image width.
                    # Weft runs horizontally: spacing measured across image height.
                    epi = calibration / warp_spacing_px
                    ppi = calibration / weft_spacing_px
                    result = calculate_cover(epi, ppi, warp_count, weft_count)

                    with right:
                        st.subheader("Detected warp / weft guides")
                        st.image(annotated, use_container_width=True)
                        st.caption("Yellow/cyan guides indicate estimated warp spacing; orange guides indicate weft spacing.")

                    st.divider()
                    st.subheader("📊 Analysis results")

                    metric_cols = st.columns(4)
                    metric_cols[0].metric("EPI", f"{epi:.2f}")
                    metric_cols[1].metric("PPI", f"{ppi:.2f}")
                    metric_cols[2].metric("Warp Cover", f"{result['warp_cover']:.2f}")
                    metric_cols[3].metric("Weft Cover", f"{result['weft_cover']:.2f}")

                    result_cols = st.columns(3)
                    result_cols[0].metric("Total Cover Factor", f"{result['total_cover']:.2f}")
                    result_cols[1].metric("Fractional Cover", f"{result['fractional_cover']:.4f}")
                    result_cols[2].metric("% Cover", f"{result['percent_cover']:.2f}%")

                    if result["classification"] == "Loose":
                        st.success(f"Fabric classification: **{result['classification']}**")
                    elif result["classification"] == "Medium":
                        st.warning(f"Fabric classification: **{result['classification']}**")
                    else:
                        st.error(f"Fabric classification: **{result['classification']}**")

                    st.caption(
                        f"Estimated spacing — warp: {warp_spacing_px:.2f} px; "
                        f"weft: {weft_spacing_px:.2f} px. Calibration: {calibration} px/in."
                    )

                    # Matplotlib chart: the classification bands and measured total.
                    fig, ax = plt.subplots(figsize=(8, 3.6))
                    band_names = ["Loose (<15)", "Medium (15–25)", "Tight (>25)"]
                    band_values = [15, 25, 35]
                    bars = ax.bar(band_names, band_values, color=["#9ecae1", "#6baed6", "#3182bd"])
                    ax.axhline(result["total_cover"], color="#d95f02", linewidth=2,
                               linestyle="--", label=f"Your fabric: {result['total_cover']:.2f}")
                    ax.set_ylabel("Cover factor")
                    ax.set_title("Cover factor classification")
                    ax.legend()
                    ax.grid(axis="y", alpha=0.2)
                    st.pyplot(fig, clear_figure=True)
                    plt.close(fig)

                    st.download_button(
                        "⬇️ Download results (CSV)",
                        data=(
                            "Parameter,Value\n"
                            f"EPI,{epi:.3f}\nPPI,{ppi:.3f}\n"
                            f"Warp Cover,{result['warp_cover']:.3f}\n"
                            f"Weft Cover,{result['weft_cover']:.3f}\n"
                            f"Total Cover Factor,{result['total_cover']:.3f}\n"
                            f"Fractional Cover,{result['fractional_cover']:.4f}\n"
                            f"Percent Cover,{result['percent_cover']:.2f}\n"
                            f"Classification,{result['classification']}\n"
                        ),
                        file_name="fabric_cover_factor_results.csv",
                        mime="text/csv",
                    )

    except Exception as error:
        st.error(f"Unable to read or process this image: {error}")

st.divider()
st.subheader("📚 Method and limitations")
st.markdown(
    """
    1. The image is converted to grayscale and contrast is enhanced using CLAHE.
    2. Sobel gradients calculate edge strength in horizontal and vertical directions.
    3. Autocorrelation estimates the repeating pixel distance between thread features.
    4. EPI and PPI are calculated from calibration divided by the estimated pixel spacing.
    5. Warp, weft and total cover factors are calculated using the Peirce equations.

    **Important for textile laboratory work:** a mobile photograph does not contain
    a physical inch scale automatically. The calibration slider must represent a
    known scale in the same plane as the fabric. The automatic algorithm is an
    estimate and can be affected by twill structure, slubs, textured yarns, moiré,
    perspective, shadows and uneven lighting. Confirm EPI/PPI with a pick glass or
    calibrated microscope before using results as laboratory measurements.

    Fractional cover in this app is defined as total cover factor divided by 28,
    following the convention requested for this project; it is not a universal
    optical area-cover measurement.
    """
)
st.caption("Fabric Cover Factor Analyzer • Streamlit • OpenCV • NumPy • Pillow")
