let buffer = [];
let session = null;

// Load ONNX model (LSTM + FC1 for JS inference)
async function loadModel() {
    session = await ort.InferenceSession.create('/static/js_model.onnx');
}

// Min-Max Normalization (same as Python)
function normalize(seq) {
    let normalized = [];
    for (let i = 0; i < 3; i++) {  // Normalize x, y, dt separately
        const col = seq.map(row => row[i]);
        const min = Math.min(...col);
        const max = Math.max(...col);

        normalized.push(col.map(val => (val - min) / (max - min + 1e-8)));
    }

    let res = [];
    for (let j = 0; j < seq.length; j++) {
        res.push([normalized[0][j], normalized[1][j], normalized[2][j]]);
    }
    return res;
}

// Run model (forward pass) in browser
async function runModel(seq) {
    const tensor = new ort.Tensor('float32', seq.flat(), [1, 20, 3]);
    const feeds = {input: tensor};
    const results = await session.run(feeds);
    return Array.from(results.output.data);  // Clean array for Flask API
}

// Send latent vector to Flask backend
async function sendToServer(latent) {
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({features: latent})
    });

    const result = await response.json();
    if (result.result === "human") {
        window.removeEventListener('mousemove', trackMouse);
        window.location.href = '/';  // Redirect to human page
    }
}

// Mouse Tracking Function
let lastTimestamp = null;
function trackMouse(e) {
    const now = performance.now();  // High precision time (ms with decimals)

    if (lastTimestamp === null) {
        lastTimestamp = now;
    }

    const dt = now - lastTimestamp;  // delta time between two points
    lastTimestamp = now;

    buffer.push([e.clientX, e.clientY, dt]);  // (x, y, dt)

    if (buffer.length === 20) {
        const normSeq = normalize(buffer);

        console.log("Raw Buffer (x, y, dt): ", buffer);
        console.log("Normalized Input: ", normSeq);

        runModel(normSeq).then(latent => sendToServer(latent));

        buffer = [];
    }
}

// On window load, load model & start capturing mouse
window.onload = async function () {
    await loadModel();
    window.addEventListener('mousemove', trackMouse);
}
