# Physics-Based Action Recognition System 

A computer vision system that uses light physics (shadow casting) to detect true depth and hand-to-face interaction using a standard 2D webcam.

## Features
- **True Depth Perception:** Uses shadow projection logic to distinguish between a hand *waving* and a hand *touching* the face.
- **Light Source Tracking:** Automatically detects the ambient light source to calculate shadow vectors.
- **Privacy First:** No images are stored; all processing is real-time.

## How it Works
Unlike standard AI that uses only geometry (X,Y coordinates), this system analyzes the **Shadow Penumbra** and **Intensity Loss** on the face. If a hand is geometrically close but casts no shadow, the system knows it is not physically touching the face.

## Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
