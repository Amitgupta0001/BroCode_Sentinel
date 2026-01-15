// BroCode Sentinel JavaScript/TypeScript SDK
// Official JavaScript client for BroCode Sentinel API

class BroCodeSDK {
    /**
     * Initialize BroCode SDK client
     * @param {string} apiKey - Your API key (starts with 'bk_')
     * @param {string} baseUrl - Base URL of BroCode Sentinel API
     */
    constructor(apiKey, baseUrl = 'http://localhost:5000') {
        this.apiKey = apiKey;
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.timeout = 30000; // 30 seconds
    }

    /**
     * Make HTTP request to API
     * @private
     */
    async _request(method, endpoint, data = null, params = null) {
        const url = new URL(`${this.baseUrl}/api/v1${endpoint}`);

        if (params) {
            Object.keys(params).forEach(key =>
                url.searchParams.append(key, params[key])
            );
        }

        const options = {
            method: method,
            headers: {
                'X-API-Key': this.apiKey,
                'Content-Type': 'application/json'
            }
        };

        if (data) {
            options.body = JSON.stringify(data);
        }

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), this.timeout);

            options.signal = controller.signal;

            const response = await fetch(url.toString(), options);
            clearTimeout(timeoutId);

            const responseData = await response.json();

            if (!response.ok) {
                throw new BroCodeAPIError(
                    responseData.error || 'API request failed',
                    response.status,
                    responseData
                );
            }

            return responseData;
        } catch (error) {
            if (error instanceof BroCodeAPIError) {
                throw error;
            }
            throw new BroCodeAPIError(
                `Request failed: ${error.message}`,
                0,
                {}
            );
        }
    }

    /**
     * Check API health status
     * @returns {Promise<Object>} Health status
     */
    async healthCheck() {
        return await this._request('GET', '/health');
    }

    /**
     * Register a new user
     * @param {string} username - Username
     * @param {string} language - Language (english, spanish, etc.)
     * @param {Array} keystrokes - Keystroke timing data
     * @param {string} email - Optional email address
     * @returns {Promise<Object>} Registration result
     */
    async registerUser(username, language, keystrokes, email = null) {
        const data = {
            username,
            language,
            keystrokes
        };

        if (email) {
            data.email = email;
        }

        return await this._request('POST', '/auth/register', data);
    }

    /**
     * Verify user authentication
     * @param {string} username - Username
     * @param {string} language - Language
     * @param {Array} keystrokes - Keystroke timing data
     * @returns {Promise<Object>} Verification result with trust score
     */
    async verifyUser(username, language, keystrokes) {
        const data = {
            username,
            language,
            keystrokes
        };

        return await this._request('POST', '/auth/verify', data);
    }

    /**
     * Get current trust score for user
     * @param {string} username - Username
     * @returns {Promise<Object>} Trust score and components
     */
    async getTrustScore(username) {
        return await this._request('GET', `/users/${username}/trust`);
    }

    /**
     * Get user's session history
     * @param {string} username - Username
     * @param {number} limit - Number of sessions to return
     * @param {number} offset - Pagination offset
     * @returns {Promise<Object>} List of sessions
     */
    async getUserSessions(username, limit = 10, offset = 0) {
        const params = { limit, offset };
        return await this._request('GET', `/users/${username}/sessions`, null, params);
    }

    /**
     * Register a webhook for events
     * @param {string} url - Webhook URL
     * @param {Array<string>} events - List of events to subscribe to
     * @param {string} secret - Optional webhook secret
     * @returns {Promise<Object>} Webhook registration result
     */
    async registerWebhook(url, events, secret = null) {
        const data = {
            url,
            events
        };

        if (secret) {
            data.secret = secret;
        }

        return await this._request('POST', '/webhooks', data);
    }

    /**
     * Get system statistics
     * @returns {Promise<Object>} System statistics
     */
    async getStats() {
        return await this._request('GET', '/stats');
    }
}

/**
 * Custom error class for API errors
 */
class BroCodeAPIError extends Error {
    constructor(message, statusCode, details) {
        super(message);
        this.name = 'BroCodeAPIError';
        this.statusCode = statusCode;
        this.details = details;
    }

    toString() {
        return `BroCodeAPIError(${this.statusCode}): ${this.message}`;
    }
}

/**
 * Create a BroCode SDK client
 * @param {string} apiKey - Your API key
 * @param {string} baseUrl - Base URL of API
 * @returns {BroCodeSDK} SDK client instance
 */
function createClient(apiKey, baseUrl = 'http://localhost:5000') {
    return new BroCodeSDK(apiKey, baseUrl);
}

// Export for Node.js/CommonJS
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        BroCodeSDK,
        BroCodeAPIError,
        createClient
    };
}

// Export for ES6 modules
if (typeof window !== 'undefined') {
    window.BroCodeSDK = BroCodeSDK;
    window.BroCodeAPIError = BroCodeAPIError;
    window.createBroCodeClient = createClient;
}

// Example usage
/*
// Initialize client
const client = new BroCodeSDK(
  'bk_your_api_key_here',
  'http://localhost:5000'
);

// Health check
try {
  const health = await client.healthCheck();
  console.log('API Status:', health.status);
} catch (error) {
  console.error('Error:', error.message);
}

// Register user
try {
  const keystrokes = [
    { key: 'a', press_time: 100, release_time: 150 },
    { key: 'b', press_time: 200, release_time: 250 }
  ];
  
  const result = await client.registerUser(
    'john_doe',
    'english',
    keystrokes,
    'john@example.com'
  );
  
  console.log('Registration:', result);
} catch (error) {
  console.error('Error:', error.message);
}

// Get trust score
try {
  const score = await client.getTrustScore('john_doe');
  console.log('Trust Score:', score.trust_score);
} catch (error) {
  console.error('Error:', error.message);
}
*/
