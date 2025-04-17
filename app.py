from flask import Flask, request, jsonify, render_template, make_response, redirect, url_for
import torch
import torch.nn as nn

app = Flask(__name__)

device = 'cpu'


# Define Final Classifier with only FC2
class FinalClassifier(nn.Module):
    def __init__(self):
        super(FinalClassifier, self).__init__()
        self.fc2 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(x))


# Load the model
model = FinalClassifier().to(device)
model.fc2.load_state_dict(torch.load('server_model.pth', map_location=device))
model.eval()


# Home Route — Show page based on cookie
@app.route("/")
def index():
    status = request.cookies.get('status', 'bot')  # default = bot
    if status == 'human':
        return render_template("human.html")
    return render_template("index.html")


# Prediction Route — Detect human or bot & Set Cookie
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json['features']
    print(request.json)
    print("abc")

    # Handle dict input case (JS might send dict sometimes)
    if isinstance(data, dict):
        data = [v for _, v in sorted(data.items(), key=lambda item: int(item[0]))]

    x = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
    prob = model(x).item()
    print("prob is:", prob)

    if prob < 0.5:
        result = "human"
    else:
        result = "bot"

    current_status = request.cookies.get('status', 'bot')

    # Prepare Response with Cookie update logic
    response = jsonify({"result": result})

    # Set or Update Cookie only if necessary
    if result == 'human' and current_status != 'human':
        response.set_cookie('status', 'human')
    if result == 'bot' and current_status == 'human':
        response.set_cookie('status', 'bot')

    return response


if __name__ == "__main__":
    app.run(debug=True)
