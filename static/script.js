const SEVERITY_OPTIONS = [
  { value: 0, label: "None" },
  { value: 1, label: "Mild" },
  { value: 2, label: "Moderate" },
  { value: 3, label: "Severe" }
];

const DISEASES = [
  {
    id: "brain_tumor",
    title: "Brain Tumor",
    navLabel: "Brain Tumor Prediction",
    mode: "image",
    endpoint: "/api/predict/brain-tumor"
  },
  {
    id: "heart",
    title: "Heart Disease",
    navLabel: "Heart Disease Prediction",
    mode: "form",
    endpoint: "/api/predict/heart",
    fields: [
      { key: "age", label: "Age", type: "number", step: "1" },
      {
        key: "sex",
        label: "Sex",
        type: "select",
        options: [
          { value: 0, label: "Female" },
          { value: 1, label: "Male" }
        ]
      },
      { key: "chest_pain", label: "Chest Pain", type: "select", options: SEVERITY_OPTIONS },
      {
        key: "shortness_of_breath",
        label: "Shortness of Breath",
        type: "select",
        options: SEVERITY_OPTIONS
      },
      { key: "fatigue", label: "Fatigue", type: "select", options: SEVERITY_OPTIONS },
      {
        key: "irregular_heartbeat",
        label: "Irregular Heartbeat / Palpitations",
        type: "select",
        options: SEVERITY_OPTIONS
      },
      {
        key: "dizziness",
        label: "Dizziness or Fainting",
        type: "select",
        options: SEVERITY_OPTIONS
      },
      {
        key: "arm_jaw_pain",
        label: "Pain in Left Arm, Shoulder, or Jaw",
        type: "select",
        options: SEVERITY_OPTIONS
      }
    ]
  },
  {
    id: "diabetes",
    title: "Diabetes",
    navLabel: "Diabetes Prediction",
    mode: "form",
    endpoint: "/api/predict/diabetes",
    fields: [
      { key: "age", label: "Age", type: "number", step: "1" },
      {
        key: "pregnancies",
        label: "Pregnancies (If Applicable)",
        type: "number",
        step: "1"
      },
      {
        key: "increased_hunger",
        label: "Increased Hunger",
        type: "select",
        options: SEVERITY_OPTIONS
      },
      {
        key: "frequent_urination",
        label: "Frequent Urination",
        type: "select",
        options: SEVERITY_OPTIONS
      },
      {
        key: "excessive_thirst",
        label: "Excessive Thirst",
        type: "select",
        options: SEVERITY_OPTIONS
      },
      {
        key: "unexplained_weight_loss",
        label: "Unexplained Weight Loss",
        type: "select",
        options: SEVERITY_OPTIONS
      },
      {
        key: "fatigue_tiredness",
        label: "Fatigue / Tiredness",
        type: "select",
        options: SEVERITY_OPTIONS
      },
      {
        key: "blurred_vision",
        label: "Blurred Vision",
        type: "select",
        options: SEVERITY_OPTIONS
      }
    ]
  }
];

let activeDiseaseId = DISEASES[0].id;

const diseaseNav = document.getElementById("diseaseNav");
const predictionPanel = document.getElementById("predictionPanel");
const historyList = document.getElementById("historyList");
const refreshHistoryButton = document.getElementById("refreshHistoryButton");

function getActiveDisease() {
  return DISEASES.find((disease) => disease.id === activeDiseaseId);
}

function renderNav() {
  diseaseNav.innerHTML = DISEASES.map(
    (disease) => `
      <button
        type="button"
        class="nav-button ${disease.id === activeDiseaseId ? "active" : ""}"
        data-disease="${disease.id}"
      >
        ${escapeHtml(disease.navLabel)}
      </button>
    `
  ).join("");
}

function renderPredictionPanel() {
  const disease = getActiveDisease();

  predictionPanel.innerHTML = `
    <div class="section-header compact-header">
      <h2>${escapeHtml(disease.title)} Prediction</h2>
    </div>

    ${disease.mode === "image" ? renderBrainUploadForm() : renderTabularForm(disease)}

    <div id="resultCard" class="result-card empty">
      <div class="result-title">Prediction Result</div>
      <p class="result-summary">Submit the form to see the prediction result.</p>
    </div>
  `;

  attachPredictionHandler(disease);
}

function renderBrainUploadForm() {
  return `
    <form id="predictionForm">
      <div class="upload-block">
        <label class="input-block">
          <span>Choose Brain MRI Scan</span>
          <input id="brainScanInput" name="scan" type="file" accept="image/*" required />
        </label>

        <img id="brainPreview" class="upload-preview" alt="Brain MRI preview" />
      </div>

      <div class="button-row">
        <button class="primary-button" type="submit">Predict Brain Tumor</button>
        <button class="secondary-button" type="reset">Reset</button>
      </div>
    </form>
  `;
}

function renderTabularForm(disease) {
  return `
    <form id="predictionForm">
      <div class="form-grid">
        ${disease.fields.map(renderField).join("")}
      </div>

      <div class="button-row">
        <button class="primary-button" type="submit">Predict ${escapeHtml(disease.title)}</button>
        <button class="secondary-button" type="reset">Reset</button>
      </div>
    </form>
  `;
}

function renderField(field) {
  if (field.type === "select") {
    return `
      <label class="input-block">
        <span>${escapeHtml(field.label)}</span>
        <select name="${escapeHtml(field.key)}" required>
          <option value="">Select</option>
          ${field.options
            .map(
              (option) =>
                `<option value="${escapeHtml(String(option.value))}">${escapeHtml(option.label)}</option>`
            )
            .join("")}
        </select>
      </label>
    `;
  }

  return `
    <label class="input-block">
      <span>${escapeHtml(field.label)}</span>
      <input
        type="number"
        name="${escapeHtml(field.key)}"
        step="${escapeHtml(field.step || "1")}"
        ${field.key === "pregnancies" ? 'value="0"' : ""}
        required
      />
    </label>
  `;
}

function attachPredictionHandler(disease) {
  const form = document.getElementById("predictionForm");
  const resultCard = document.getElementById("resultCard");

  if (disease.mode === "image") {
    const fileInput = document.getElementById("brainScanInput");
    const preview = document.getElementById("brainPreview");

    fileInput.addEventListener("change", () => {
      const [file] = fileInput.files;
      if (!file) {
        preview.style.display = "none";
        preview.removeAttribute("src");
        return;
      }

      preview.src = URL.createObjectURL(file);
      preview.style.display = "block";
    });
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setResultLoading(resultCard, disease.title);

    try {
      const result =
        disease.mode === "image"
          ? await submitBrainTumorPrediction(form, disease)
          : await submitTabularPrediction(form, disease);

      renderPredictionResult(resultCard, result);
      await loadHistory();
    } catch (error) {
      renderError(resultCard, error.message || "Prediction request failed.");
    }
  });

  form.addEventListener("reset", () => {
    window.setTimeout(() => {
      resultCard.className = "result-card empty";
      resultCard.innerHTML = `
        <div class="result-title">Prediction Result</div>
        <p class="result-summary">Submit the form to see the prediction result.</p>
      `;

      const preview = document.getElementById("brainPreview");
      if (preview) {
        preview.style.display = "none";
        preview.removeAttribute("src");
      }
    }, 0);
  });
}

async function submitTabularPrediction(form, disease) {
  const formData = new FormData(form);
  const payload = {};

  disease.fields.forEach((field) => {
    payload[field.key] =
      field.type === "number" ? Number(formData.get(field.key)) : Number(formData.get(field.key));
  });

  const response = await fetch(disease.endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  return handleJsonResponse(response);
}

async function submitBrainTumorPrediction(form, disease) {
  const formData = new FormData(form);
  const file = formData.get("scan");

  if (!file || !file.name) {
    throw new Error("Please upload a brain MRI image first.");
  }

  const uploadData = new FormData();
  uploadData.append("scan", file);

  const response = await fetch(disease.endpoint, {
    method: "POST",
    body: uploadData
  });

  return handleJsonResponse(response);
}

async function handleJsonResponse(response) {
  const data = await response.json().catch(() => ({}));

  if (response.status === 401) {
    window.location.href = "/login";
    throw new Error("Please log in again.");
  }

  if (!response.ok) {
    throw new Error(data.error || `Request failed with status ${response.status}.`);
  }

  return data;
}

function setResultLoading(resultCard, title) {
  resultCard.className = "result-card empty";
  resultCard.innerHTML = `
    <div class="result-title">Prediction Result</div>
    <p class="result-summary">Running ${escapeHtml(title.toLowerCase())} prediction...</p>
  `;
}

function renderPredictionResult(resultCard, result) {
  const details = buildResultDetails(result);
  const imageBlock = result.result_image_data_url
    ? `
        <div class="result-image-block">
          <img class="result-image" src="${result.result_image_data_url}" alt="Brain MRI result" />
          ${result.localization_note ? `<p class="helper-text">${escapeHtml(result.localization_note)}</p>` : ""}
        </div>
      `
    : "";
  const disclaimerBlock = result.disclaimer
    ? `<p class="helper-text">${escapeHtml(result.disclaimer)}</p>`
    : "";

  resultCard.className = "result-card";
  resultCard.innerHTML = `
    <div class="result-title">Prediction Result</div>
    <p class="result-summary">${escapeHtml(result.result_text)}</p>
    <div class="detail-grid">
      ${details
        .map(
          (item) => `
            <div class="detail-card">
              <strong>${escapeHtml(item.label)}</strong>
              <p>${escapeHtml(item.value)}</p>
            </div>
          `
        )
        .join("")}
    </div>
    ${imageBlock}
    ${disclaimerBlock}
  `;
}

function renderError(resultCard, message) {
  resultCard.className = "result-card error";
  resultCard.innerHTML = `
    <div class="result-title">Prediction Result</div>
    <p class="result-summary">${escapeHtml(message)}</p>
  `;
}

function buildResultDetails(result) {
  const details = [];

  if (result.predicted_class_display) {
    details.push({ label: "Predicted Class", value: result.predicted_class_display });
  }

  if (result.predicted_label_text) {
    details.push({ label: "Prediction", value: result.predicted_label_text });
  }

  if (typeof result.confidence === "number") {
    details.push({ label: "Confidence", value: `${result.confidence.toFixed(2)}%` });
  }

  if (result.positive_probability !== undefined) {
    details.push({
      label: "Positive Probability",
      value: `${Number(result.positive_probability).toFixed(2)}%`
    });
  }

  if (result.negative_probability !== undefined) {
    details.push({
      label: "Negative Probability",
      value: `${Number(result.negative_probability).toFixed(2)}%`
    });
  }

  return details;
}

async function loadHistory() {
  if (!historyList) {
    return;
  }

  historyList.innerHTML = `<div class="history-empty">Loading history...</div>`;

  try {
    const response = await fetch("/api/predictions/history");
    const data = await handleJsonResponse(response);
    renderHistory(data.predictions || []);
  } catch (error) {
    historyList.innerHTML = `<div class="history-empty">${escapeHtml(error.message || "Could not load history.")}</div>`;
  }
}

function renderHistory(predictions) {
  if (!predictions.length) {
    historyList.innerHTML = `<div class="history-empty">No prediction history yet.</div>`;
    return;
  }

  historyList.innerHTML = predictions
    .map((prediction) => {
      const diseaseName = formatDiseaseName(prediction.disease_key);
      const confidence =
        typeof prediction.confidence === "number"
          ? `${Number(prediction.confidence).toFixed(2)}%`
          : "Not available";

      return `
        <article class="history-item">
          <div class="history-meta-row">
            <strong>${escapeHtml(diseaseName)}</strong>
            <span>${escapeHtml(formatDate(prediction.created_at))}</span>
          </div>
          <p>${escapeHtml(prediction.result_text || "No result text available.")}</p>
          <div class="history-submeta">
            <span>${escapeHtml(prediction.predicted_label_text || prediction.predicted_class_display || "Prediction recorded")}</span>
            <span>Confidence: ${escapeHtml(confidence)}</span>
          </div>
        </article>
      `;
    })
    .join("");
}

function formatDiseaseName(value) {
  if (value === "brain_tumor") {
    return "Brain Tumor";
  }
  if (value === "heart") {
    return "Heart Disease";
  }
  if (value === "diabetes") {
    return "Diabetes";
  }
  return value;
}

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

diseaseNav.addEventListener("click", (event) => {
  const button = event.target.closest("[data-disease]");
  if (!button) {
    return;
  }

  activeDiseaseId = button.dataset.disease;
  renderNav();
  renderPredictionPanel();
});

if (refreshHistoryButton) {
  refreshHistoryButton.addEventListener("click", loadHistory);
}

renderNav();
renderPredictionPanel();
loadHistory();
