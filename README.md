# Fabric Cover Factor Analyzer

A Streamlit application that estimates EPI, PPI and Peirce fabric cover factor from a close-up woven-fabric photograph.

## Project structure

```text
fabric_cover_factor_streamlit/
├── app.py
├── requirements.txt
└── README.md
```

## Run locally

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install packages:

```bash
pip install -r requirements.txt
```

Start Streamlit:

```bash
streamlit run app.py
```

Open the local URL printed in the terminal (usually `http://localhost:8501`).

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and `README.md`.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select **Create app**, choose the repository and branch.
4. Set the main file path to `app.py`.
5. Deploy. Streamlit Cloud installs the packages from `requirements.txt`.

No database or secrets are required.

## How the image analysis works

1. Pillow loads the uploaded image and corrects EXIF orientation.
2. OpenCV converts it to grayscale, applies CLAHE contrast enhancement and Gaussian blur.
3. Sobel X/Y gradients produce directional edge-energy profiles.
4. Autocorrelation estimates the repeating pixel spacing of yarn features.
5. Calibration converts pixel spacing to threads per inch:
   - `EPI = pixels_per_inch / warp_spacing_pixels`
   - `PPI = pixels_per_inch / weft_spacing_pixels`
6. The app calculates cover factors using the equations below.

The image orientation assumption is that warp yarns run vertically and weft yarns horizontally. Rotate the image before upload if required.

## Equations

```text
Warp Cover = EPI / sqrt(Warp Count Ne)
Weft Cover = PPI / sqrt(Weft Count Ne)
Total Cover = Warp Cover + Weft Cover
              - (Warp Cover * Weft Cover) / 28

Fractional Cover = Total Cover / 28
Percentage Cover = Fractional Cover * 100
```

Classification:
- Loose: total cover factor < 15
- Medium: 15 <= total cover factor <= 25
- Tight: total cover factor > 25

## Calibration and scientific limitations

A photograph does not intrinsically contain physical inch information. The calibration slider must be set using a known physical reference measured in the same plane as the fabric (for example, a scale photographed beside the fabric without changing camera distance or plane).

The image algorithm estimates periodic spacing. It does not reliably identify every individual yarn centre. Twill, satin, textured yarns, slubs, high-density fabrics, moiré, shadows, blur and perspective distortion can reduce accuracy. Validate EPI/PPI with a pick glass or calibrated microscope for formal textile testing.

Fractional cover is normalized by 28 as requested in this project; this should not be confused with a universal optical area-cover measurement.

## Group Members / Project By

- 24-23TE055
- 24TE103
- 24TE052
