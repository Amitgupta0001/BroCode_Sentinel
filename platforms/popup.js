// BroCode Sentinel - Popup Script
// Handles popup UI interactions

// DOM Elements
const loggedOutView = document.getElementById('logged-out');
const loggedInView = document.getElementById('logged-in');
const loadingView = document.getElementById('loading');
const errorView = document.getElementById('error');

const loginForm = document.getElementById('login-form');
const usernameInput = document.getElementById('username');
const apiKeyInput = document.getElementById('apiKey');

const userName = document.getElementById('user-name');
const userStatus = document.getElementById('user-status');
const trustScoreValue = document.getElementById('trust-score');
const trustLevel = document.getElementById('trust-level');
const lastCheckValue = document.getElementById('last-check');
const sessionAgeValue = document.getElementById('session-age');

const refreshBtn = document.getElementById('refresh-btn');
const settingsBtn = document.getElementById('settings-btn');
const logoutBtn = document.getElementById('logout-btn');

// Initialize popup
async function init() {
    showLoading();

    try {
        // Get current session
        const response = await sendMessage({ action: 'getSession' });

        if (response.session) {
            showLoggedIn(response.session);
            await updateTrustScore();
        } else {
            showLoggedOut();
        }
    } catch (error) {
        showError(error.message);
    }
}

// Show loading state
function showLoading() {
    hideAll();
    loadingView.classList.remove('hidden');
}

// Show logged out view
function showLoggedOut() {
    hideAll();
    loggedOutView.classList.remove('hidden');
}

// Show logged in view
function showLoggedIn(session) {
    hideAll();
    loggedInView.classList.remove('hidden');

    userName.textContent = session.username;
    updateSessionAge(session.loginTime);
}

// Hide all views
function hideAll() {
    loggedOutView.classList.add('hidden');
    loggedInView.classList.add('hidden');
    loadingView.classList.add('hidden');
    errorView.classList.add('hidden');
}

// Show error
function showError(message) {
    errorView.textContent = message;
    errorView.classList.remove('hidden');

    setTimeout(() => {
        errorView.classList.add('hidden');
    }, 5000);
}

// Send message to background script
function sendMessage(message) {
    return new Promise((resolve, reject) => {
        chrome.runtime.sendMessage(message, (response) => {
            if (response.error) {
                reject(new Error(response.error));
            } else {
                resolve(response);
            }
        });
    });
}

// Handle login
loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    showLoading();

    try {
        const username = usernameInput.value.trim();
        const apiKey = apiKeyInput.value.trim();

        if (!username || !apiKey) {
            throw new Error('Please enter username and API key');
        }

        // Save API key
        await chrome.storage.local.set({ apiKey });

        // Login
        const response = await sendMessage({
            action: 'login',
            data: {
                username,
                language: 'english',
                keystrokes: [] // Would collect actual keystrokes
            }
        });

        if (response.success) {
            showLoggedIn(response.session);
            await updateTrustScore();
        }
    } catch (error) {
        showLoggedOut();
        showError(error.message);
    }
});

// Handle logout
logoutBtn.addEventListener('click', async () => {
    try {
        await sendMessage({ action: 'logout' });
        showLoggedOut();
    } catch (error) {
        showError(error.message);
    }
});

// Handle refresh
refreshBtn.addEventListener('click', async () => {
    refreshBtn.textContent = '🔄 Refreshing...';
    refreshBtn.disabled = true;

    try {
        await sendMessage({ action: 'checkTrustScore' });
        await updateTrustScore();
    } catch (error) {
        showError(error.message);
    } finally {
        refreshBtn.textContent = '🔄 Refresh Score';
        refreshBtn.disabled = false;
    }
});

// Handle settings
settingsBtn.addEventListener('click', () => {
    chrome.runtime.openOptionsPage();
});

// Update trust score display
async function updateTrustScore() {
    try {
        const response = await sendMessage({ action: 'getTrustScore' });

        const score = response.trustScore || 0;
        const percentage = Math.round(score * 100);

        trustScoreValue.textContent = `${percentage}%`;

        // Update trust level
        let level, levelClass;
        if (score >= 0.8) {
            level = 'Excellent';
            levelClass = 'excellent';
        } else if (score >= 0.6) {
            level = 'Good';
            levelClass = 'good';
        } else if (score >= 0.4) {
            level = 'Warning';
            levelClass = 'warning';
        } else {
            level = 'Critical';
            levelClass = 'critical';
        }

        trustLevel.textContent = level;
        trustLevel.className = `trust-level ${levelClass}`;

        // Update last check time
        if (response.lastCheck) {
            const elapsed = Date.now() - response.lastCheck;
            const minutes = Math.floor(elapsed / 60000);

            if (minutes < 1) {
                lastCheckValue.textContent = 'Just now';
            } else if (minutes === 1) {
                lastCheckValue.textContent = '1 min ago';
            } else {
                lastCheckValue.textContent = `${minutes} mins ago`;
            }
        }
    } catch (error) {
        console.error('Failed to update trust score:', error);
    }
}

// Update session age
function updateSessionAge(loginTime) {
    const update = () => {
        const elapsed = Date.now() - loginTime;
        const hours = Math.floor(elapsed / 3600000);
        const minutes = Math.floor((elapsed % 3600000) / 60000);

        if (hours > 0) {
            sessionAgeValue.textContent = `${hours}h ${minutes}m`;
        } else {
            sessionAgeValue.textContent = `${minutes}m`;
        }
    };

    update();
    setInterval(update, 60000); // Update every minute
}

// Initialize on load
document.addEventListener('DOMContentLoaded', init);

// Auto-refresh trust score every 30 seconds
setInterval(() => {
    if (!loggedInView.classList.contains('hidden')) {
        updateTrustScore();
    }
}, 30000);
