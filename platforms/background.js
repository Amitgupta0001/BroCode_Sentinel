// BroCode Sentinel - Background Service Worker
// Handles background tasks, API communication, and monitoring

// Configuration
const CONFIG = {
    apiUrl: 'http://localhost:5000',
    checkInterval: 60000, // 1 minute
    sessionTimeout: 3600000 // 1 hour
};

// State
let currentSession = null;
let trustScore = 0;
let lastCheck = 0;

// Initialize extension
chrome.runtime.onInstalled.addListener(() => {
    console.log('BroCode Sentinel extension installed');

    // Set up periodic checks
    chrome.alarms.create('securityCheck', { periodInMinutes: 1 });

    // Load saved session
    loadSession();
});

// Load session from storage
async function loadSession() {
    const result = await chrome.storage.local.get(['session', 'apiKey']);
    if (result.session) {
        currentSession = result.session;
        console.log('Session loaded:', currentSession.username);
    }
    if (result.apiKey) {
        CONFIG.apiKey = result.apiKey;
    }
}

// Save session to storage
async function saveSession(session) {
    currentSession = session;
    await chrome.storage.local.set({ session });
}

// Clear session
async function clearSession() {
    currentSession = null;
    await chrome.storage.local.remove('session');
}

// API Request helper
async function apiRequest(endpoint, method = 'GET', data = null) {
    const url = `${CONFIG.apiUrl}/api/v1${endpoint}`;

    const options = {
        method,
        headers: {
            'Content-Type': 'application/json'
        }
    };

    if (CONFIG.apiKey) {
        options.headers['X-API-Key'] = CONFIG.apiKey;
    }

    if (data) {
        options.body = JSON.stringify(data);
    }

    try {
        const response = await fetch(url, options);
        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'API request failed');
        }

        return result;
    } catch (error) {
        console.error('API request error:', error);
        throw error;
    }
}

// Check trust score
async function checkTrustScore() {
    if (!currentSession || !currentSession.username) {
        return;
    }

    try {
        const result = await apiRequest(`/users/${currentSession.username}/trust`);

        trustScore = result.trust_score;
        lastCheck = Date.now();

        // Update badge
        updateBadge(trustScore);

        // Check for alerts
        if (trustScore < 0.5) {
            showNotification(
                'Security Alert',
                `Your trust score is low (${(trustScore * 100).toFixed(0)}%). Please re-authenticate.`,
                'warning'
            );
        }

        // Store trust score
        await chrome.storage.local.set({ trustScore, lastCheck });

    } catch (error) {
        console.error('Failed to check trust score:', error);
    }
}

// Update extension badge
function updateBadge(score) {
    const percentage = Math.round(score * 100);

    // Set badge text
    chrome.action.setBadgeText({ text: `${percentage}` });

    // Set badge color based on score
    let color;
    if (score >= 0.8) {
        color = '#10b981'; // Green
    } else if (score >= 0.5) {
        color = '#f59e0b'; // Orange
    } else {
        color = '#ef4444'; // Red
    }

    chrome.action.setBadgeBackgroundColor({ color });
}

// Show notification
function showNotification(title, message, type = 'info') {
    const iconUrl = type === 'warning' ? 'icons/warning.png' : 'icons/icon48.png';

    chrome.notifications.create({
        type: 'basic',
        iconUrl,
        title,
        message,
        priority: type === 'warning' ? 2 : 1
    });
}

// Monitor active tab
chrome.tabs.onActivated.addListener(async (activeInfo) => {
    const tab = await chrome.tabs.get(activeInfo.tabId);

    if (currentSession) {
        // Log tab activity
        console.log('Tab activated:', tab.url);

        // Check for sensitive sites
        if (isSensitiveSite(tab.url)) {
            // Verify trust score
            await checkTrustScore();

            if (trustScore < 0.7) {
                showNotification(
                    'Security Warning',
                    'You are accessing a sensitive site with a low trust score. Please re-authenticate.',
                    'warning'
                );
            }
        }
    }
});

// Check if site is sensitive
function isSensitiveSite(url) {
    const sensitiveDomains = [
        'bank',
        'paypal',
        'stripe',
        'admin',
        'dashboard',
        'account'
    ];

    return sensitiveDomains.some(domain => url.includes(domain));
}

// Handle alarms
chrome.alarms.onAlarm.addListener((alarm) => {
    if (alarm.name === 'securityCheck') {
        checkTrustScore();
    }
});

// Handle messages from popup/content scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    switch (request.action) {
        case 'login':
            handleLogin(request.data)
                .then(sendResponse)
                .catch(error => sendResponse({ error: error.message }));
            return true;

        case 'logout':
            handleLogout()
                .then(sendResponse)
                .catch(error => sendResponse({ error: error.message }));
            return true;

        case 'getTrustScore':
            sendResponse({ trustScore, lastCheck });
            return true;

        case 'checkTrustScore':
            checkTrustScore()
                .then(() => sendResponse({ trustScore }))
                .catch(error => sendResponse({ error: error.message }));
            return true;

        case 'getSession':
            sendResponse({ session: currentSession });
            return true;

        default:
            sendResponse({ error: 'Unknown action' });
    }
});

// Handle login
async function handleLogin(credentials) {
    try {
        // Verify credentials with API
        const result = await apiRequest('/auth/verify', 'POST', credentials);

        if (result.verified) {
            const session = {
                username: credentials.username,
                loginTime: Date.now(),
                trustScore: result.trust_score
            };

            await saveSession(session);
            trustScore = result.trust_score;
            updateBadge(trustScore);

            showNotification(
                'Login Successful',
                `Welcome back, ${credentials.username}!`
            );

            return { success: true, session };
        } else {
            throw new Error('Authentication failed');
        }
    } catch (error) {
        throw error;
    }
}

// Handle logout
async function handleLogout() {
    await clearSession();
    trustScore = 0;
    chrome.action.setBadgeText({ text: '' });

    showNotification(
        'Logged Out',
        'You have been logged out successfully'
    );

    return { success: true };
}

// Auto-logout on session timeout
setInterval(() => {
    if (currentSession) {
        const sessionAge = Date.now() - currentSession.loginTime;

        if (sessionAge > CONFIG.sessionTimeout) {
            handleLogout();
            showNotification(
                'Session Expired',
                'Your session has expired. Please log in again.',
                'warning'
            );
        }
    }
}, 60000); // Check every minute

console.log('BroCode Sentinel background service worker loaded');
