from flask import Flask, request, jsonify, render_template, make_response
import math

app = Flask(__name__)

# Fixed weights and bias
w1 = 2.2862465
w2 = -1.8418641
b = -5.2830777

def predict_manual(x1, x2):
    prod1 = w1 * x1
    prod2 = w2 * x2
    z = prod1 + prod2 + b
    prob = 1 / (1 + math.exp(-z))
    print(f"x1={x1}, x2={x2}, prod1={prod1}, prod2={prod2}, z={z}, prob={prob}")
    return "human" if prob < 0.5 else "bot"

@app.route("/")
def index():
    status = request.cookies.get('status', 'bot') 
    if status == 'human':
        return render_template("human.html")
    return render_template("index.html")

# Prediction Route — Detect human or bot & Set Cookie
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json['features']
    print("Received features:", data)

    # Handle dict input case
    if isinstance(data, dict):
        data = [v for _, v in sorted(data.items(), key=lambda item: int(item[0]))]

    x1, x2 = float(data[0]), float(data[1])  # Ensure numeric values
    result = predict_manual(x1, x2)

    current_status = request.cookies.get('status', 'bot')
    response = jsonify({"result": result})

    # Set or update cookie only if necessary
    if result == 'human' and current_status != 'human':
        response.set_cookie('status', 'human')
    elif result == 'bot' and current_status == 'human':
        response.set_cookie('status', 'bot')

    return response

if __name__ == "__main__":
    app.run(debug=True)
