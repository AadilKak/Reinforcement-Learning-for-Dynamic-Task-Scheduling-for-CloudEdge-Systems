from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
import numpy as np
import os

# Import your models
from dqn import DQNNetwork
from ppo import PPONetwork

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Configuration === #
MODEL_PATH = os.getenv("RL_MODEL_PATH", "models/dqn_cloud_edge.pt")
ALGORITHM = os.getenv("RL_ALGORITHM", "DQN")  # Options: DQN or PPO
STATE_DIM = int(os.getenv("STATE_DIM", 25))
ACTION_DIM = int(os.getenv("ACTION_DIM", 10))

# === Load Model === #
model = None
@app.before_request
def load_model():
    global model
    if ALGORITHM.upper() == "DQN":
        model = DQNNetwork(STATE_DIM, ACTION_DIM).to(device)
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['policy_net'])
    elif ALGORITHM.upper() == "PPO":
        model = PPONetwork(STATE_DIM, ACTION_DIM).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        raise ValueError(f"Unsupported algorithm: {ALGORITHM}")
    model.eval()
    print(f"Loaded {ALGORITHM.upper()} model from {MODEL_PATH}")

# === Inference Endpoint === #
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if "state" not in data:
        return jsonify({"error": "Missing 'state' field"}), 400

    try:
        state = np.array(data["state"], dtype=np.float32)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            if ALGORITHM.upper() == "DQN":
                q_values = model(state_tensor)
                action = torch.argmax(q_values, dim=1).item()
            else:
                logits, _ = model(state_tensor)
                probs = F.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=1).item()

        return jsonify({"action": action})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Health Check === #
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "online", "algorithm": ALGORITHM, "model_path": MODEL_PATH})

# === Start Server === #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
