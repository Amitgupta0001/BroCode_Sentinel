// BroCode Sentinel - Content Script
// Runs on all web pages to monitor security and provide warnings

// Configuration
const SENSITIVE_KEYWORDS = [
    'password', 'credit card', 'ssn', 'social security',
    'bank account', 'routing number', 'cvv', 'pin'
];

// Monitor form inputs for sensitive data
function monitorForms() {
    const forms = document.querySelectorAll('form');

    forms.forEach(form => {
        form.addEventListener('submit', async (e) => {
            // Check if form contains sensitive data
            const inputs = form.querySelectorAll('input');
            let hasSensitiveData = false;

            inputs.forEach(input => {
                const type = input.type.toLowerCase();
                const name = input.name.toLowerCase();
                const id = input.id.toLowerCase();

                if (type === 'password' ||
                    SENSITIVE_KEYWORDS.some(keyword =>
                        name.includes(keyword) || id.includes(keyword)
                    )) {
                    hasSensitiveData = true;
                }
            });

            if (hasSensitiveData) {
                // Check trust score before submitting sensitive data
                const response = await chrome.runtime.sendMessage({
                    action: 'getTrustScore'
                });

                if (response.trustScore < 0.7) {
                    e.preventDefault();

                    const proceed = confirm(
                        '⚠️ BroCode Sentinel Security Warning\n\n' +
                        `Your current trust score is ${Math.round(response.trustScore * 100)}%.\n` +
                        'Submitting sensitive data with a low trust score is not recommended.\n\n' +
                        'Do you want to proceed anyway?'
                    );

                    if (proceed) {
                        form.submit();
                    }
                }
            }
        });
    });
}

// Detect password fields
function detectPasswordFields() {
    const passwordFields = document.querySelectorAll('input[type="password"]');

    passwordFields.forEach(field => {
        // Add visual indicator
        field.style.borderLeft = '3px solid #667eea';
        field.title = 'Protected by BroCode Sentinel';
    });
}

// Monitor clipboard for sensitive data
document.addEventListener('paste', async (e) => {
    const clipboardData = e.clipboardData.getData('text');

    // Check if clipboard contains sensitive patterns
    const patterns = [
        /\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}/, // Credit card
        /\d{3}-\d{2}-\d{4}/, // SSN
        /\d{9,16}/ // Bank account
    ];

    const hasSensitiveData = patterns.some(pattern => pattern.test(clipboardData));

    if (hasSensitiveData) {
        const response = await chrome.runtime.sendMessage({
            action: 'getTrustScore'
        });

        if (response.trustScore < 0.6) {
            e.preventDefault();

            alert(
                '⚠️ BroCode Sentinel Security Warning\n\n' +
                'You are pasting potentially sensitive data with a low trust score.\n' +
                'This action has been blocked for your security.'
            );
        }
    }
});

// Add security indicator to page
function addSecurityIndicator() {
    const indicator = document.createElement('div');
    indicator.id = 'brocode-security-indicator';
    indicator.style.cssText = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 50px;
    height: 50px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 24px;
    cursor: pointer;
    z-index: 999999;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    transition: transform 0.3s;
  `;
    indicator.innerHTML = '🛡️';
    indicator.title = 'BroCode Sentinel - Click for details';

    indicator.addEventListener('mouseenter', () => {
        indicator.style.transform = 'scale(1.1)';
    });

    indicator.addEventListener('mouseleave', () => {
        indicator.style.transform = 'scale(1)';
    });

    indicator.addEventListener('click', () => {
        chrome.runtime.sendMessage({ action: 'getTrustScore' }, (response) => {
            const score = Math.round((response.trustScore || 0) * 100);
            alert(
                `🛡️ BroCode Sentinel\n\n` +
                `Trust Score: ${score}%\n` +
                `Status: ${score >= 70 ? 'Secure' : 'Warning'}\n\n` +
                `Click the extension icon for more details.`
            );
        });
    });

    document.body.appendChild(indicator);
}

// Initialize content script
function init() {
    console.log('BroCode Sentinel content script loaded');

    // Monitor forms
    monitorForms();

    // Detect password fields
    detectPasswordFields();

    // Add security indicator
    if (document.body) {
        addSecurityIndicator();
    } else {
        document.addEventListener('DOMContentLoaded', addSecurityIndicator);
    }

    // Re-scan for new forms (for SPAs)
    const observer = new MutationObserver(() => {
        monitorForms();
        detectPasswordFields();
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
}

// Run initialization
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
