const API_URL = "http://127.0.0.1:8001/predict";
const NORMAL_URL = "http://127.0.0.1:8001/sample";
const ATTACK_URL = "http://127.0.0.1:8001/attack-sample";

async function runPrediction(url) {
    const result = document.getElementById("result");

    result.innerHTML = "⏳ Getting data...";

    try {
        // Get sample data
        const sampleResponse = await fetch(url);
        const sampleData = await sampleResponse.json();

        if (!sampleResponse.ok) {
            throw new Error("Could not get sample data");
        }

        result.innerHTML = "⏳ AI is predicting...";

        // Send the 77 features to the model
        const predictionResponse = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                features: sampleData.features
            })
        });

        const data = await predictionResponse.json();

        if (!predictionResponse.ok) {
            throw new Error(data.detail || "Prediction failed");
        }

        const probability = (data.probability * 100).toFixed(2);

        result.innerHTML = `
            <h2>🛡️ Prediction Result</h2>
            <p><strong>Actual Type:</strong> ${
                sampleData.actual_label || "Normal"
            }</p>
            <p><strong>AI Prediction:</strong> ${data.prediction}</p>
            <p><strong>Probability:</strong> ${probability}%</p>
        `;

    } catch (error) {
        result.innerHTML = `
            <h2>❌ Error</h2>
            <p>${error.message}</p>
        `;
    }
}

// Normal traffic button
document.getElementById("normalBtn").addEventListener("click", () => {
    runPrediction(NORMAL_URL);
});

// DDoS attack button
document.getElementById("attackBtn").addEventListener("click", () => {
    runPrediction(ATTACK_URL);
});
// DoS Hulk attack button
document.getElementById("dosBtn").addEventListener("click", () => {
    runPrediction("http://127.0.0.1:8001/dos-sample");
});
document.getElementById("portscanBtn").addEventListener("click", () => {
    runPrediction("http://127.0.0.1:8001/portscan-sample");
});