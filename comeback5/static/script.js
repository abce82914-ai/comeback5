const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const registerBtn = document.getElementById("registerBtn");
const recognizeBtn = document.getElementById("recognizeBtn");
const stopRecBtn = document.getElementById("stopRecBtn");
const nameInput = document.getElementById("nameInput");
const messageDiv = document.getElementById("message");

canvas.width = 640;
canvas.height = 480;

let recognizing = false;
let processing = false;
let latestFaces = [];

function showMessage(msg, isError = false) {
  messageDiv.textContent = msg;
  messageDiv.style.color = isError ? "red" : "green";
}

async function setupCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = stream;
    await video.play();
  } catch (err) {
    showMessage("Error accessing camera: " + err.message, true);
  }
}

// Capture resized frame as base64 jpeg (downscale to 320x240)
function captureResizedFrame() {
  const offscreen = document.createElement("canvas");
  offscreen.width = 320;
  offscreen.height = 240;
  const offCtx = offscreen.getContext("2d");
  offCtx.drawImage(video, 0, 0, offscreen.width, offscreen.height);
  return offscreen.toDataURL("image/jpeg", 0.7); // quality 0.7 to reduce size
}

// Draw current frame + latest face boxes
function drawFrame() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  latestFaces.forEach(face => {
    const [x, y, w, h] = face.box;

    ctx.strokeStyle = "lime";
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);

    ctx.fillStyle = "lime";
    ctx.font = "20px Arial";
    const textWidth = ctx.measureText(face.name).width;
    const textHeight = 22;
    ctx.fillRect(x, y - textHeight, textWidth + 10, textHeight);

    ctx.fillStyle = "#000";
    ctx.fillText(face.name, x + 5, y - 5);
  });
}

// Recognition loop using requestAnimationFrame and promise chaining
async function recognitionLoop() {
  if (!recognizing) {
    return;
  }

  if (!processing) {
    processing = true;

    try {
      const imgData = captureResizedFrame();

      const response = await fetch("/recognize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imgData }),
      });

      const result = await response.json();

      if (result.success) {
        // Scale face box coordinates back to 640x480 from 320x240
        latestFaces = result.faces.map(face => {
          const scaleX = canvas.width / 320;
          const scaleY = canvas.height / 240;
          const [x, y, w, h] = face.box;
          return {
            box: [
              Math.round(x * scaleX),
              Math.round(y * scaleY),
              Math.round(w * scaleX),
              Math.round(h * scaleY),
            ],
            name: face.name,
          };
        });
        showMessage("Recognizing...");
      } else {
        latestFaces = [];
        showMessage(result.message, true);
      }
    } catch (e) {
      latestFaces = [];
      showMessage("Recognition error: " + e.message, true);
    }

    processing = false;
  }

  drawFrame();
  requestAnimationFrame(recognitionLoop);
}

recognizeBtn.addEventListener("click", () => {
  if (recognizing) return;
  recognizing = true;
  recognizeBtn.disabled = true;
  stopRecBtn.disabled = false;
  showMessage("Recognition started. Faces will be detected automatically.");
  recognitionLoop();
});

stopRecBtn.addEventListener("click", () => {
  if (!recognizing) return;
  recognizing = false;
  latestFaces = [];
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  recognizeBtn.disabled = false;
  stopRecBtn.disabled = true;
  showMessage("Recognition stopped.");
});

// Register face (same as before)
registerBtn.addEventListener("click", async () => {
  const name = nameInput.value.trim();
  if (!name) {
    showMessage("Please enter a name to register.", true);
    return;
  }

  const imgData = captureResizedFrame();

  registerBtn.disabled = true;
  showMessage("Registering face...");

  try {
    const response = await fetch("/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: name, image: imgData }),
    });

    const result = await response.json();
    if (result.success) {
      showMessage(result.message);
      nameInput.value = "";
    } else {
      showMessage(result.message, true);
    }
  } catch (e) {
    showMessage("Error during registration: " + e.message, true);
  } finally {
    registerBtn.disabled = false;
  }
});

setupCamera().then(() => {
  video.addEventListener("play", () => {
    drawFrame();
  });
});
