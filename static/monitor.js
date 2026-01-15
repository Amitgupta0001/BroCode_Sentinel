// Video capture globals
const video = document.createElement("video");
const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");
let stream = null;

async function initWebcam() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    await video.play();
    console.log("Webcam initialized for value monitoring.");
  } catch (err) {
    console.warn("Webcam not available:", err);
  }
}

// Function to capture frame as Base64
function captureFrame() {
  if (!stream || video.paused || video.ended) return null;
  // Set canvas size to video size (downscale for performance if needed)
  if (canvas.width !== video.videoWidth) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  // Draw video frame to canvas
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  // Return Base64 JPEG (0.5 quality to save bandwidth)
  return canvas.toDataURL("image/jpeg", 0.5);
}

(function () {
  // Buffer of keystroke events
  let keystrokes = [];

  // Capture keystrokes globally
  document.addEventListener("keydown", function (e) {
    keystrokes.push({ key: e.key, t: Date.now(), type: "keydown" });
  });

  document.addEventListener("keyup", function (e) {
    keystrokes.push({ key: e.key, t: Date.now(), type: "keyup" });
  });

  // Optionally send small device info
  function getDeviceInfo() {
    return {
      ua: navigator.userAgent || "",
      language: navigator.language || "",
      platform: navigator.platform || ""
    };
  }

  // Prepare payload and send to /monitor_activity
  async function sendBehaviorData() {
    const userElem = document.getElementById("user") || document.getElementById("username") || {};
    const user_id = userElem.value || userElem.getAttribute && userElem.getAttribute("value") || "guest";

    // Capture real frame
    const imageData = captureFrame();

    // UI Feedback: Show "Scanning" pulse
    const pill = document.querySelector('.status-pill');
    if (pill) {
      const originalText = pill.innerText;
      pill.innerText = "Scanning...";
      setTimeout(() => { if (pill) pill.innerText = originalText; }, 1000);
    }

    const payload = {
      user_id: user_id,
      keystrokes: keystrokes,
      frame_data: {
        // We can still send placeholders if needed, but backend prefers 'image_data' now
      },
      image_data: imageData,
      device_info: getDeviceInfo()
    };

    // Only send if we have meaningful data (keystrokes OR image)
    if (keystrokes.length === 0 && !imageData) return;

    try {
      const res = await fetch("/monitor_activity", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        if (res.status === 401) {
          window.location.href = "/";
        }
        console.error("monitor_activity returned", res.status);
        return;
      }

      const data = await res.json();

      // Handle redirect (critical state - forced logout)
      if (data.redirect) {
        window.location.href = data.redirect;
        return;
      }

      // Handle progressive re-authentication
      if (data.requires_reauth && data.auth_state === "warning") {
        showReauthModal(data.reauth_deadline);
      } else {
        hideReauthModal();
      }

      // Update UI - ensure these elements exist in your dashboard/index templates
      const trustEl = document.getElementById("trust") || document.getElementById("trust_score");
      const riskEl = document.getElementById("risk") || document.getElementById("risk_status");

      if (trustEl) trustEl.innerText = (typeof data.trust_score !== "undefined") ? data.trust_score.toFixed(3) : "N/A";
      if (riskEl) riskEl.innerText = data.risk || (data.anomaly ? "high" : "normal");

      // After successful send, clear buffer
      keystrokes = [];
    } catch (err) {
      console.error("Error sending behavior data:", err);
    }
  }

  // Re-authentication modal functions
  let reauthTimer = null;

  function showReauthModal(deadline) {
    let modal = document.getElementById("reauth-modal");
    if (!modal) {
      // Create modal if it doesn't exist
      modal = document.createElement("div");
      modal.id = "reauth-modal";
      modal.className = "reauth-modal";
      modal.innerHTML = `
        <div class="reauth-content glass-panel">
          <h2>⚠️ Re-Authentication Required</h2>
          <p>Your trust score has dropped. Please verify your identity to continue.</p>
          <div class="countdown">Time remaining: <span id="reauth-countdown">30</span>s</div>
          <div class="reauth-actions">
            <button onclick="quickReauth()" class="btn-primary">Verify Now</button>
            <button onclick="dismissReauth()" class="btn-secondary">Dismiss</button>
          </div>
        </div>
      `;
      document.body.appendChild(modal);
    }

    modal.style.display = "flex";

    // Start countdown
    const now = Date.now() / 1000;
    const remaining = Math.max(0, deadline - now);
    updateCountdown(remaining);
  }

  function hideReauthModal() {
    const modal = document.getElementById("reauth-modal");
    if (modal) {
      modal.style.display = "none";
    }
    if (reauthTimer) {
      clearInterval(reauthTimer);
      reauthTimer = null;
    }
  }

  function updateCountdown(seconds) {
    const countdownEl = document.getElementById("reauth-countdown");
    if (!countdownEl) return;

    if (reauthTimer) clearInterval(reauthTimer);

    let remaining = Math.floor(seconds);
    countdownEl.innerText = remaining;

    reauthTimer = setInterval(() => {
      remaining--;
      if (remaining <= 0) {
        clearInterval(reauthTimer);
        window.location.href = "/";
      } else {
        countdownEl.innerText = remaining;
      }
    }, 1000);
  }

  window.quickReauth = function () {
    // Trigger quick re-auth (type passphrase)
    const passphrase = prompt("Type your passphrase to re-authenticate:");
    if (passphrase) {
      // Simulate typing the passphrase to generate keystrokes
      // In production, you'd have a proper modal with input field
      alert("Re-authentication successful! (Demo)");
      hideReauthModal();
    }
  };

  window.dismissReauth = function () {
    hideReauthModal();
  };

  // Send every 5 seconds (continuous authentication)
  const INTERVAL_MS = 5000;
  setInterval(sendBehaviorData, INTERVAL_MS);

  // Initialize camera
  initWebcam();

  // Expose for debugging
  window.__brocode_monitor = {
    getBuffer: () => keystrokes.slice(),
    flushNow: sendBehaviorData
  };
})();