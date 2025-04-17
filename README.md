# 🖱️ Lightweight Bot Detection using Mouse Movement

A smart, scalable, and ultra-lightweight proof-of-concept that detects bots using only mouse telemetry and **5 simple math operations** — no need for server-side deep learning.

**By Saksham Lakhera**

---

## Project Overview

This system distinguishes between human and bot users based on their natural mouse movements. Unlike heavy ML models that run on the server, we perform most computations in the browser and only send **2 latent values** to the backend.

### What It Does

- Tracks mouse telemetry (x, y, dt) in the browser.
- Runs a compressed model in JavaScript using ONNX.
- Sends latent vector `[x1, x2]` to Flask backend.
- Backend uses **5 simple math operations** to predict: **human or bot**.
- Sets a cookie based on result — updates view accordingly.

---

## Folder Structure

```
project/
│
├── app.py              # PyTorch-based server (needs server_model.pth)
├── app_simple.py       # Lightweight Flask server using basic math only
├── requirements.txt    # Python dependencies
├── server_model.pth    # Final FC2 PyTorch model used by app.py
├── static/
│   ├── script.js       # JavaScript: ONNX inference + mouse tracking
│   └── js_model.onnx   # ONNX model used in browser (LSTM + FC1)
├── templates/
│   ├── index.html      # Default page for bot users
│   └── human.html      # Page shown to humans after detection
└── analysisFiles/      # (Not used in prod) Jupyter Notebooks for training/testing
    ├── mouse.ipynb
    ├── scrapper.ipynb
    └── weightExtractor.ipynb
```

---

## How to Run

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ `app.py` requires PyTorch. If you're running `app_simple.py`, you **don't need** PyTorch or `server_model.pth`.

---

### Step 2: Run the Flask Server

#### Option A: With PyTorch model

```bash
python app.py
```

> Needs `server_model.pth`

#### Option B: Lightweight math-only version (language-agnostic logic)

```bash
python app_simple.py
```

> Great reference if you want to implement backend in Rust, Java, C++, etc.

---

### Step 3: Open in Browser

Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

Move your mouse around — if your movements are natural, you’ll be redirected to a human-only page (`human.html`)!

---

## Requirements Summary

- Python ≥ 3.8
- Flask
- For ML version: PyTorch
- For analysis: Jupyter, pandas, matplotlib, torch, etc. (see `requirements.txt`)

---

## How It Works

1. **Client (JS)**:
   - Tracks 20 mouse movements as `[x, y, dt]`
   - Normalizes sequence
   - Runs ONNX model in-browser → gets `[x1, x2]`
   - Sends `[x1, x2]` to backend

2. **Server**:
   - `app.py`: Uses PyTorch model to compute sigmoid
   - `app_simple.py`: Uses this equation:

     ```python
     z = w1 * x1 + w2 * x2 + b
     prob = 1 / (1 + math.exp(-z))
     ```

   - Result < 0.5 → Human  
     Result ≥ 0.5 → Bot

---

## Why This is Smart

- No PyTorch/TF on server
- Super scalable — handles millions of requests with ease
- Browser does the heavy lifting
- Minimal data transfer
- Language-agnostic backend logic

---

## For Training / Analysis

Refer to files in `analysisFiles/` folder:

- `mouse.ipynb` — sequence generation and normalization
- `weightExtractor.ipynb` — extracts weights for app_simple.py
- `scrapper.ipynb` — dataset collection (if applicable)

These are **not required** to run the detection server.

---
## 📖 Read More

For a detailed walkthrough and explanation, check out the full blog post on Medium:  
👉 [Built a Bot Detection System with Just 5 Simple Calculations](https://medium.com/@sakshamlakhera/built-a-bot-detection-system-with-just-5-simple-calculations-8409717da585)

