from flask import Flask, request, jsonify, render_template_string
from predict import predict
from datetime import datetime

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>

<meta charset="UTF-8">
<title>AI Medical Diagnosis System</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family: 'Poppins', sans-serif;
}

body {
    min-height: 100vh;
    background: linear-gradient(-45deg, #e8f0ff, #ffffff, #dbe9ff, #ffffff);
    background-size: 400% 400%;
    animation: gradientMove 12s ease infinite;
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.container {
    max-width: 1100px;
    margin: auto;
    padding: 40px 20px;
}

.header {
    text-align: center;
    margin-bottom: 40px;
}

.header h1 {
    font-size: 42px;
    color: #1f4e79;
    font-weight: 700;
}

.header p {
    color: #555;
    margin-top: 10px;
    font-size: 16px;
}

.card {
    background: rgba(255,255,255,0.9);
    backdrop-filter: blur(10px);
    padding: 30px;
    border-radius: 18px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 25px;
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-4px);
}

textarea {
    width: 100%;
    height: 120px;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #ccc;
    font-size: 16px;
    resize: none;
    margin-top: 12px;
}

button {
    width: 100%;
    margin-top: 20px;
    padding: 14px;
    border: none;
    border-radius: 12px;
    background: linear-gradient(90deg, #1f77d0, #3fa0ff);
    color: white;
    font-size: 18px;
    font-weight: 600;
    cursor: pointer;
    transition: 0.3s;
}

button:hover {
    transform: scale(1.02);
    box-shadow: 0 6px 15px rgba(0,0,0,0.15);
}

.result-card {
    background: #f8fbff;
    padding: 20px;
    border-radius: 14px;
    margin-top: 15px;
    border-left: 6px solid #1f77d0;
    animation: fadeIn 0.5s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

.progress-bar {
    height: 12px;
    background: #e0e0e0;
    border-radius: 10px;
    margin-top: 12px;
    overflow: hidden;
}

.progress {
    height: 100%;
    background: linear-gradient(90deg, #1f77d0, #3fa0ff);
    width: 0%;
    transition: width 0.6s ease;
}

.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    background: #1f77d0;
    color: white;
    margin-left: 10px;
}

.loading {
    text-align: center;
    margin-top: 15px;
    font-weight: 500;
    color: #1f77d0;
}

.spinner {
    margin: auto;
    margin-top: 10px;
    width: 28px;
    height: 28px;
    border: 4px solid #e0e0e0;
    border-top: 4px solid #1f77d0;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.footer {
    text-align: center;
    margin-top: 40px;
    color: #777;
    font-size: 14px;
}

</style>
</head>

<body>

<div class="container">

<div class="header">

<h1>🩺 AI Medical Diagnosis System</h1>

<p>
Deep Learning Based Disease Prediction using LSTM Neural Network
</p>

</div>

<div class="card">

<label>
<strong>Enter Patient Symptoms</strong>
</label>

<textarea
id="symptoms"
placeholder="Example: fever cough headache fatigue"
></textarea>

<button onclick="predictDisease()">
🔍 Predict Disease
</button>

<div id="loading" class="loading"></div>

</div>

<div class="card">

<h2>Prediction Results</h2>

<div id="results"></div>

</div>

<div class="footer">

AI Medical Diagnoser • Generated at {{ time }}

</div>

</div>

<script>

async function predictDisease() {

const symptoms =
document.getElementById("symptoms").value;

if (symptoms.trim() === "") {

alert("Please enter symptoms");
return;

}

document.getElementById("loading").innerHTML =

"Analyzing symptoms using AI model..."
+
"<div class='spinner'></div>";

try {

const response = await fetch(
"/predict",
{
method: "POST",
headers: {
"Content-Type": "application/json"
},
body: JSON.stringify({
symptoms: symptoms
})
}
);

const data =
await response.json();

document.getElementById("loading").innerHTML = "";

displayResults(data);

}

catch {

document.getElementById("loading").innerText =
"Server connection error";

}

}

function displayResults(results) {

const resultsDiv =
document.getElementById("results");

resultsDiv.innerHTML = "";

results.forEach((item, index) => {

const percent =
(item.confidence * 100).toFixed(2);

const badge =
index === 0
? "<span class='badge'>Most Likely</span>"
: "";

const card =
document.createElement("div");

card.className =
"result-card";

card.innerHTML =

`
<h3>
${item.disease}
${badge}
</h3>

<p>
Confidence: ${percent}%
</p>

<div class="progress-bar">
<div
class="progress"
style="width:${percent}%"
></div>
</div>
`;

resultsDiv.appendChild(card);

});

}

</script>

</body>
</html>
"""

@app.route("/")
def home():

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return render_template_string(
        HTML_PAGE,
        time=current_time
    )

@app.route("/predict", methods=["POST"])
def predict_api():

    data = request.get_json()

    symptoms = data.get("symptoms", "")

    results = predict(symptoms)

    response = []

    for disease, confidence in results:

        response.append({
            "disease": disease,
            "confidence": float(confidence)
        })

    return jsonify(response)

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
