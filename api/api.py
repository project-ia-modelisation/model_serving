from flask import Flask, request, jsonify
from script.evaluate import evaluate_model, load_preprocessed_model
from script.generate import generate_and_save_model

app = Flask(__name__)

@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.json
    model_path = data.get("model_path")
    ground_truth_path = data.get("ground_truth_path")

    model = load_preprocessed_model(model_path)
    ground_truth = load_preprocessed_model(ground_truth_path)
    result = evaluate_model(model, ground_truth)
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
