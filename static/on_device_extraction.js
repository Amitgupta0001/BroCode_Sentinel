// On-Device Feature Extraction Module
// Extracts features on client-side for privacy and performance

class OnDeviceFeatureExtractor {
    constructor() {
        this.initialized = false;
    }

    /**
     * Initialize feature extractor
     */
    async init() {
        console.log('Initializing on-device feature extraction...');
        this.initialized = true;
        return true;
    }

    /**
     * Extract facial features from video frame
     * @param {HTMLVideoElement|HTMLCanvasElement} source - Video or canvas element
     * @returns {Object} Extracted features
     */
    async extractFacialFeatures(source) {
        try {
            // Create canvas for processing
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');

            // Set canvas size
            canvas.width = source.videoWidth || source.width || 640;
            canvas.height = source.videoHeight || source.height || 480;

            // Draw frame
            ctx.drawImage(source, 0, 0, canvas.width, canvas.height);

            // Get image data
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

            // Extract features
            const features = {
                // Basic image statistics
                brightness: this._calculateBrightness(imageData),
                contrast: this._calculateContrast(imageData),
                sharpness: this._calculateSharpness(imageData),

                // Face presence indicators (simplified)
                face_present: this._detectFacePresence(imageData),
                face_size_ratio: this._estimateFaceSize(imageData),
                face_position: this._estimateFacePosition(imageData),

                // Quality metrics
                lighting_quality: this._assessLighting(imageData),
                image_quality: this._assessImageQuality(imageData),

                // Metadata
                timestamp: Date.now(),
                frame_size: {
                    width: canvas.width,
                    height: canvas.height
                }
            };

            return features;
        } catch (error) {
            console.error('Error extracting facial features:', error);
            return null;
        }
    }

    /**
     * Extract keystroke features
     * @param {Array} keystrokes - Array of keystroke events
     * @returns {Object} Extracted features
     */
    extractKeystrokeFeatures(keystrokes) {
        if (!keystrokes || keystrokes.length < 2) {
            return null;
        }

        const features = {
            // Hold times (press to release)
            hold_times: [],
            hold_time_mean: 0,
            hold_time_std: 0,

            // Flight times (release to next press)
            flight_times: [],
            flight_time_mean: 0,
            flight_time_std: 0,

            // Typing speed
            typing_speed: 0,

            // Rhythm variance
            rhythm_variance: 0,

            // Error indicators
            backspace_count: 0,
            error_rate: 0,

            // Metadata
            total_keys: keystrokes.length,
            timestamp: Date.now()
        };

        // Calculate hold times
        keystrokes.forEach(ks => {
            if (ks.press_time && ks.release_time) {
                const holdTime = ks.release_time - ks.press_time;
                features.hold_times.push(holdTime);
            }

            // Count backspaces
            if (ks.key && ks.key.toLowerCase() === 'backspace') {
                features.backspace_count++;
            }
        });

        // Calculate flight times
        for (let i = 0; i < keystrokes.length - 1; i++) {
            if (keystrokes[i].release_time && keystrokes[i + 1].press_time) {
                const flightTime = keystrokes[i + 1].press_time - keystrokes[i].release_time;
                features.flight_times.push(flightTime);
            }
        }

        // Calculate statistics
        if (features.hold_times.length > 0) {
            features.hold_time_mean = this._mean(features.hold_times);
            features.hold_time_std = this._std(features.hold_times);
        }

        if (features.flight_times.length > 0) {
            features.flight_time_mean = this._mean(features.flight_times);
            features.flight_time_std = this._std(features.flight_times);
            features.rhythm_variance = features.flight_time_std;
        }

        // Calculate typing speed (characters per minute)
        if (keystrokes.length >= 2) {
            const firstTime = keystrokes[0].press_time || 0;
            const lastTime = keystrokes[keystrokes.length - 1].press_time || 0;
            const durationMinutes = (lastTime - firstTime) / 60000;

            if (durationMinutes > 0) {
                features.typing_speed = keystrokes.length / durationMinutes;
            }
        }

        // Calculate error rate
        features.error_rate = features.backspace_count / keystrokes.length;

        return features;
    }

    /**
     * Extract device context features
     * @returns {Object} Device context
     */
    extractDeviceContext() {
        return {
            // Screen information
            screen_width: window.screen.width,
            screen_height: window.screen.height,
            screen_color_depth: window.screen.colorDepth,

            // Viewport information
            viewport_width: window.innerWidth,
            viewport_height: window.innerHeight,

            // Device capabilities
            has_camera: navigator.mediaDevices ? true : false,
            has_microphone: navigator.mediaDevices ? true : false,

            // Browser information
            user_agent: navigator.userAgent,
            language: navigator.language,
            platform: navigator.platform,

            // Time information
            timezone_offset: new Date().getTimezoneOffset(),
            timestamp: Date.now()
        };
    }

    /**
     * Calculate brightness of image
     * @private
     */
    _calculateBrightness(imageData) {
        const data = imageData.data;
        let sum = 0;

        for (let i = 0; i < data.length; i += 4) {
            // Average RGB values
            sum += (data[i] + data[i + 1] + data[i + 2]) / 3;
        }

        return sum / (data.length / 4) / 255; // Normalize to 0-1
    }

    /**
     * Calculate contrast of image
     * @private
     */
    _calculateContrast(imageData) {
        const data = imageData.data;
        const brightness = [];

        for (let i = 0; i < data.length; i += 4) {
            brightness.push((data[i] + data[i + 1] + data[i + 2]) / 3);
        }

        return this._std(brightness) / 255; // Normalize to 0-1
    }

    /**
     * Calculate sharpness using Laplacian
     * @private
     */
    _calculateSharpness(imageData) {
        // Simplified sharpness calculation
        // In production, use proper Laplacian operator
        const contrast = this._calculateContrast(imageData);
        return Math.min(1.0, contrast * 2);
    }

    /**
     * Detect face presence (simplified)
     * @private
     */
    _detectFacePresence(imageData) {
        // Simplified: check if there's a concentrated area of skin tones
        // In production, use proper face detection
        const brightness = this._calculateBrightness(imageData);
        return brightness > 0.3 && brightness < 0.8;
    }

    /**
     * Estimate face size ratio
     * @private
     */
    _estimateFaceSize(imageData) {
        // Simplified estimation
        // In production, use actual face detection
        return 0.25; // Assume 25% of frame
    }

    /**
     * Estimate face position
     * @private
     */
    _estimateFacePosition(imageData) {
        // Simplified: assume center
        return {
            x: 0.5,
            y: 0.5
        };
    }

    /**
     * Assess lighting quality
     * @private
     */
    _assessLighting(imageData) {
        const brightness = this._calculateBrightness(imageData);

        // Optimal brightness: 0.4 - 0.7
        if (brightness >= 0.4 && brightness <= 0.7) {
            return 1.0;
        } else if (brightness < 0.2 || brightness > 0.9) {
            return 0.3;
        } else {
            return 0.7;
        }
    }

    /**
     * Assess overall image quality
     * @private
     */
    _assessImageQuality(imageData) {
        const brightness = this._calculateBrightness(imageData);
        const contrast = this._calculateContrast(imageData);
        const sharpness = this._calculateSharpness(imageData);

        // Weighted average
        return (brightness * 0.3 + contrast * 0.3 + sharpness * 0.4);
    }

    /**
     * Calculate mean of array
     * @private
     */
    _mean(arr) {
        if (arr.length === 0) return 0;
        return arr.reduce((a, b) => a + b, 0) / arr.length;
    }

    /**
     * Calculate standard deviation
     * @private
     */
    _std(arr) {
        if (arr.length === 0) return 0;
        const mean = this._mean(arr);
        const squaredDiffs = arr.map(x => Math.pow(x - mean, 2));
        return Math.sqrt(this._mean(squaredDiffs));
    }

    /**
     * Compress features for transmission
     * @param {Object} features - Features to compress
     * @returns {Object} Compressed features
     */
    compressFeatures(features) {
        // Round numbers to reduce size
        const compressed = {};

        for (const [key, value] of Object.entries(features)) {
            if (typeof value === 'number') {
                compressed[key] = Math.round(value * 1000) / 1000;
            } else if (Array.isArray(value)) {
                // Only send statistics, not raw arrays
                compressed[`${key}_mean`] = Math.round(this._mean(value) * 1000) / 1000;
                compressed[`${key}_std`] = Math.round(this._std(value) * 1000) / 1000;
            } else {
                compressed[key] = value;
            }
        }

        return compressed;
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = OnDeviceFeatureExtractor;
}
