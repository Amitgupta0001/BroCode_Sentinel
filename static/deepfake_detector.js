// Deepfake Detection Module
// Detects AI-generated faces and video spoofing

class DeepfakeDetector {
    constructor() {
        this.isInitialized = false;
        this.frameHistory = [];
        this.maxHistoryLength = 30; // 30 frames for temporal analysis
    }

    async initialize() {
        try {
            this.isInitialized = true;
            console.log('Deepfake detector initialized');
            return true;
        } catch (err) {
            console.error('Deepfake detector initialization failed:', err);
            return false;
        }
    }

    async analyzeFrame(videoElement, canvas) {
        /**
         * Multi-method deepfake detection:
         * 1. Frequency domain analysis
         * 2. Blink pattern analysis
         * 3. Texture consistency
         * 4. Temporal consistency
         */

        const ctx = canvas.getContext('2d');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        ctx.drawImage(videoElement, 0, 0);

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        // Store frame for temporal analysis
        this.frameHistory.push({
            timestamp: Date.now(),
            imageData: imageData
        });

        if (this.frameHistory.length > this.maxHistoryLength) {
            this.frameHistory.shift();
        }

        // Run detection methods
        const scores = {
            frequency: this.frequencyAnalysis(imageData),
            blink: this.blinkAnalysis(),
            texture: this.textureAnalysis(imageData),
            temporal: this.temporalConsistency()
        };

        // Weighted fusion
        const finalScore = (
            scores.frequency * 0.3 +
            scores.blink * 0.3 +
            scores.texture * 0.2 +
            scores.temporal * 0.2
        );

        return {
            isDeepfake: finalScore > 0.5,
            confidence: finalScore,
            scores: scores,
            message: finalScore > 0.5 ? 'Deepfake detected' : 'Real face detected'
        };
    }

    frequencyAnalysis(imageData) {
        /**
         * Deepfakes often have artifacts in frequency domain
         * Analyze high-frequency components
         */

        const data = imageData.data;
        const width = imageData.width;
        const height = imageData.height;

        // Convert to grayscale
        const grayscale = new Float32Array(width * height);
        for (let i = 0; i < data.length; i += 4) {
            const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
            grayscale[i / 4] = gray;
        }

        // Compute high-frequency energy
        let highFreqEnergy = 0;
        const sampleSize = Math.min(1000, grayscale.length - 1);

        for (let i = 0; i < sampleSize; i++) {
            const idx = Math.floor(Math.random() * (grayscale.length - 1));
            const diff = Math.abs(grayscale[idx + 1] - grayscale[idx]);
            highFreqEnergy += diff;
        }

        highFreqEnergy /= sampleSize;

        // Deepfakes tend to have lower high-frequency energy (smoother)
        // Normalize: lower energy = higher deepfake score
        const normalizedScore = Math.max(0, Math.min(1, (30 - highFreqEnergy) / 30));

        return normalizedScore;
    }

    blinkAnalysis() {
        /**
         * Analyze blink patterns
         * Deepfakes often have unnatural blink rates or durations
         */

        if (this.frameHistory.length < 10) {
            return 0; // Not enough data
        }

        // Simplified blink detection based on brightness changes
        const recentFrames = this.frameHistory.slice(-10);
        let blinkCount = 0;
        let lastBrightness = this.getAverageBrightness(recentFrames[0].imageData);

        for (let i = 1; i < recentFrames.length; i++) {
            const brightness = this.getAverageBrightness(recentFrames[i].imageData);
            const diff = Math.abs(brightness - lastBrightness);

            // Significant brightness drop might indicate blink
            if (diff > 20) {
                blinkCount++;
            }

            lastBrightness = brightness;
        }

        // Normal blink rate: 15-20 per minute
        // In 10 frames (~2 seconds at 5fps), expect 0-1 blinks
        // Too many or too few blinks is suspicious
        const expectedBlinks = 0.5;
        const deviation = Math.abs(blinkCount - expectedBlinks);

        return Math.min(1, deviation / 2);
    }

    textureAnalysis(imageData) {
        /**
         * Analyze texture consistency
         * Deepfakes may have inconsistent skin texture
         */

        const data = imageData.data;

        // Sample random patches and compute texture variance
        const numPatches = 10;
        const patchSize = 20;
        const variances = [];

        for (let p = 0; p < numPatches; p++) {
            const x = Math.floor(Math.random() * (imageData.width - patchSize));
            const y = Math.floor(Math.random() * (imageData.height - patchSize));

            const patchVariance = this.computePatchVariance(data, x, y, patchSize, imageData.width);
            variances.push(patchVariance);
        }

        // Compute variance of variances
        const mean = variances.reduce((a, b) => a + b, 0) / variances.length;
        const varianceOfVariances = variances.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / variances.length;

        // High variance of variances suggests inconsistent texture (deepfake)
        const normalizedScore = Math.min(1, varianceOfVariances / 1000);

        return normalizedScore;
    }

    temporalConsistency() {
        /**
         * Analyze temporal consistency across frames
         * Deepfakes may have temporal artifacts
         */

        if (this.frameHistory.length < 5) {
            return 0;
        }

        const recentFrames = this.frameHistory.slice(-5);
        let inconsistencies = 0;

        for (let i = 1; i < recentFrames.length; i++) {
            const prev = recentFrames[i - 1].imageData;
            const curr = recentFrames[i].imageData;

            // Sample random pixels and compute difference
            const sampleSize = 100;
            let totalDiff = 0;

            for (let s = 0; s < sampleSize; s++) {
                const idx = Math.floor(Math.random() * (prev.data.length / 4)) * 4;

                const diff = Math.abs(prev.data[idx] - curr.data[idx]) +
                    Math.abs(prev.data[idx + 1] - curr.data[idx + 1]) +
                    Math.abs(prev.data[idx + 2] - curr.data[idx + 2]);

                totalDiff += diff;
            }

            const avgDiff = totalDiff / sampleSize;

            // Large sudden changes might indicate temporal inconsistency
            if (avgDiff > 50) {
                inconsistencies++;
            }
        }

        // Normalize
        return Math.min(1, inconsistencies / (recentFrames.length - 1));
    }

    getAverageBrightness(imageData) {
        const data = imageData.data;
        let sum = 0;
        const sampleSize = Math.min(1000, data.length / 4);

        for (let i = 0; i < sampleSize; i++) {
            const idx = Math.floor(Math.random() * (data.length / 4)) * 4;
            const brightness = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
            sum += brightness;
        }

        return sum / sampleSize;
    }

    computePatchVariance(data, x, y, size, width) {
        const values = [];

        for (let dy = 0; dy < size; dy++) {
            for (let dx = 0; dx < size; dx++) {
                const idx = ((y + dy) * width + (x + dx)) * 4;
                const gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
                values.push(gray);
            }
        }

        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;

        return variance;
    }

    reset() {
        this.frameHistory = [];
    }
}

// Export for use in other modules
window.DeepfakeDetector = DeepfakeDetector;
