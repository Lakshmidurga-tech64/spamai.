/* ------------------------------------------------------------
   NexGen — AI Voice Security
   Handles file selection, drag & drop, the API call to the
   FastAPI backend, and rendering the result on screen.
------------------------------------------------------------- */

// Change this if your backend runs on a different host/port.
const API_BASE_URL = "http://127.0.0.1:8000";

// ---- Element references ----
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const fileInfo = document.getElementById("file-info");
const fileNameEl = document.getElementById("file-name");
const clearFileBtn = document.getElementById("clear-file");
const analyzeBtn = document.getElementById("analyze-btn");
const errorBox = document.getElementById("error-box");

const uploadPanel = document.getElementById("upload-panel");
const loadingPanel = document.getElementById("loading-panel");
const resultPanel = document.getElementById("result-panel");

const demoBanner = document.getElementById("demo-banner");
const predictionEl = document.getElementById("result-prediction");
const confidenceEl = document.getElementById("result-confidence");
const riskScoreEl = document.getElementById("result-risk-score");
const riskLevelEl = document.getElementById("result-risk-level");
const meterFill = document.getElementById("meter-fill");
const explanationEl = document.getElementById("result-explanation");
const securityIcon = document.getElementById("security-icon");
const securityMessage = document.getElementById("security-message");
const resetBtn = document.getElementById("reset-btn");

const ALLOWED_EXTENSIONS = [".wav", ".mp3", ".m4a"];
const MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024; // 15MB, matches backend

let selectedFile = null;

// ---------------- File selection ----------------

function getExtension(filename){
  const dotIndex = filename.lastIndexOf(".");
  return dotIndex === -1 ? "" : filename.slice(dotIndex).toLowerCase();
}

function showError(message){
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function clearError(){
  errorBox.hidden = true;
  errorBox.textContent = "";
}

function handleFileSelected(file){
  clearError();

  if (!file){
    return;
  }

  const extension = getExtension(file.name);
  if (!ALLOWED_EXTENSIONS.includes(extension)){
    showError("Unsupported file type. Please upload a WAV, MP3, or M4A file.");
    return;
  }

  if (file.size > MAX_FILE_SIZE_BYTES){
    showError("File is too large. Please upload a file under 15MB.");
    return;
  }

  selectedFile = file;
  fileNameEl.textContent = file.name;
  fileInfo.hidden = false;
  analyzeBtn.disabled = false;
}

function clearSelectedFile(){
  selectedFile = null;
  fileInput.value = "";
  fileInfo.hidden = true;
  analyzeBtn.disabled = true;
  clearError();
}

fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  handleFileSelected(file);
});

clearFileBtn.addEventListener("click", clearSelectedFile);

// ---- Drag and drop ----

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("drag-over");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("drag-over");
  });
});

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  handleFileSelected(file);
});

// ---------------- Analyze ----------------

function setPanelState(state){
  // state: "upload" | "loading" | "result"
  uploadPanel.hidden = state !== "upload";
  loadingPanel.hidden = state !== "loading";
  resultPanel.hidden = state !== "result";
}

function riskClass(riskLevel){
  if (riskLevel === "LOW") return "risk-low";
  if (riskLevel === "MEDIUM") return "risk-medium";
  return "risk-high";
}

function riskColor(riskLevel){
  if (riskLevel === "LOW") return "#34d98a";
  if (riskLevel === "MEDIUM") return "#f2b84b";
  return "#ef5a6f";
}

function renderResult(data){
  demoBanner.hidden = !data.demo_mode;

  predictionEl.textContent = data.prediction;
  predictionEl.className = "result-value prediction-value " + riskClass(data.risk_level);

  confidenceEl.textContent = `${data.confidence}%`;
  riskScoreEl.textContent = `${data.risk_score}/100`;

  riskLevelEl.textContent = data.risk_level;
  riskLevelEl.className = "result-value " + riskClass(data.risk_level);

  meterFill.style.width = `${data.risk_score}%`;
  meterFill.style.background = riskColor(data.risk_level);

  explanationEl.textContent = data.explanation;

  securityIcon.textContent = data.risk_level === "HIGH" ? "⚠" : "ℹ";
  securityMessage.textContent = data.message;

  setPanelState("result");
}

async function analyzeVoice(){
  if (!selectedFile){
    showError("No file selected. Please choose an audio file first.");
    return;
  }

  clearError();
  setPanelState("loading");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try{
    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok){
      let detailMessage = "Unable to analyze this audio. Please upload a valid WAV, MP3, or M4A file.";
      try{
        const errorData = await response.json();
        if (errorData && errorData.detail){
          detailMessage = errorData.detail;
        }
      } catch(_parseError){
        // response wasn't JSON; keep the default message
      }
      setPanelState("upload");
      showError(detailMessage);
      return;
    }

    const data = await response.json();
    renderResult(data);

  } catch(networkError){
    setPanelState("upload");
    showError("Backend unavailable. Please make sure the server is running and try again.");
  }
}

analyzeBtn.addEventListener("click", analyzeVoice);

// ---------------- Reset ----------------

resetBtn.addEventListener("click", () => {
  clearSelectedFile();
  setPanelState("upload");
});
