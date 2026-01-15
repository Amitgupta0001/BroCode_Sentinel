// Liveness Detection Module
// Prevents spoofing attacks using photos, videos, or masks

class LivenessDetector {
    constructor() {
        this.challenges = [
            { type: 'blink', instruction: 'Please blink twice', duration: 3000 },
            { type: 'smile', instruction: 'Please smile', duration: 2000 },
            { type: 'turn_left', instruction: 'Turn your head left', duration: 2000 },
            { type: 'turn_right', instruction: 'Turn your head right', duration: 2000 },
            { type: 'nod', instruction: 'Nod your head', duration: 2000 }
        ];

        this.currentChallenge = null;
        this.challengeStartTime = null;
        this.blinkCount = 0;
        this.lastBlinkTime = 0;
        this.faceDetector = null;
        this.isActive = false;
    }

    async initialize() {
        try {
            // Load face detection model (using face-api.js or similar)
            // For now, we'll use basic detection
            this.isActive = true;
            console.log('Liveness detector initialized');
            return true;
        } catch (err) {
            console.error('Liveness detector initialization failed:', err);
            return false;
        }
    }

    startRandomChallenge() {
        // Select random challenge
        const randomIndex = Math.floor(Math.random() * this.challenges.length);
        this.currentChallenge = this.challenges[randomIndex];
        this.challengeStartTime = Date.now();
        this.blinkCount = 0;

        return {
            type: this.currentChallenge.type,
            instruction: this.currentChallenge.instruction,
            duration: this.currentChallenge.duration,
            startTime: this.challengeStartTime
        };
    }

    async analyzeFrame(videoElement, canvas) {
        if (!this.currentChallenge) return null;

        const ctx = canvas.getContext('2d');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        ctx.drawImage(videoElement, 0, 0);

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        // Analyze based on challenge type
        const result = await this.analyzeChallengeResponse(imageData);

        return result;
    }

    async analyzeChallengeResponse(imageData) {
        const elapsed = Date.now() - this.challengeStartTime;
        const challengeType = this.currentChallenge.type;

        // Simplified analysis (in production, use ML models)
        const analysis = {
            type: challengeType,
            elapsed: elapsed,
            completed: false,
            confidence: 0,
            details: {}
        };

        switch (challengeType) {
            case 'blink':
                analysis.completed = this.detectBlink(imageData);
                analysis.confidence = this.blinkCount >= 2 ? 0.9 : 0.3;
                analysis.details = { blinkCount: this.blinkCount };
                break;

            case 'smile':
                analysis.completed = this.detectSmile(imageData);
                analysis.confidence = analysis.completed ? 0.85 : 0.2;
                break;

            case 'turn_left':
            case 'turn_right':
                analysis.completed = this.detectHeadTurn(imageData, challengeType);
                analysis.confidence = analysis.completed ? 0.8 : 0.2;
                break;

            case 'nod':
                analysis.completed = this.detectNod(imageData);
                analysis.confidence = analysis.completed ? 0.8 : 0.2;
                break;
        }

        // Check if challenge timed out
        if (elapsed > this.currentChallenge.duration) {
            analysis.timedOut = true;
            if (!analysis.completed) {
                analysis.confidence = 0;
            }
        }

        return analysis;
    }

    detectBlink(imageData) {
        // Simplified blink detection
        // In production, use eye aspect ratio (EAR) from facial landmarks

        // Simulate blink detection based on brightness changes
        const now = Date.now();
        const avgBrightness = this.calculateAverageBrightness(imageData);

        // If brightness drops significantly (eyes closed)
        if (avgBrightness < 100 && now - this.lastBlinkTime > 300) {
            this.blinkCount++;
            this.lastBlinkTime = now;
            console.log(`Blink detected! Count: ${this.blinkCount}`);
        }

        return this.blinkCount >= 2;
    }

    detectSmile(imageData) {
        // Simplified smile detection
        // In production, use facial landmark analysis

        // Simulate smile detection
        const mouthRegion = this.extractMouthRegion(imageData);
        const mouthWidth = this.calculateMouthWidth(mouthRegion);

        // If mouth is wider than threshold, consider it a smile
        return mouthWidth > 0.6;
    }

    detectHeadTurn(imageData, direction) {
        // Simplified head turn detection
        // In production, use head pose estimation

        const faceCenter = this.detectFaceCenter(imageData);
        const imageCenter = imageData.width / 2;

        if (direction === 'turn_left') {
            return faceCenter < imageCenter * 0.7;
        } else {
            return faceCenter > imageCenter * 1.3;
        }
    }

    detectNod(imageData) {
        // Simplified nod detection
        // In production, use vertical head movement tracking

        // Simulate nod detection
        return Math.random() > 0.5; // Placeholder
    }

    calculateAverageBrightness(imageData) {
        let sum = 0;
        const data = imageData.data;

        // Sample every 10th pixel for performance
        for (let i = 0; i < data.length; i += 40) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            sum += (r + g + b) / 3;
        }

        return sum / (data.length / 40);
    }

    extractMouthRegion(imageData) {
        // Simplified mouth region extraction
        // In production, use facial landmarks
        const height = imageData.height;
        const width = imageData.width;

        return {
            x: width * 0.3,
            y: height * 0.6,
            width: width * 0.4,
            height: height * 0.2
        };
    }

    calculateMouthWidth(mouthRegion) {
        // Simplified mouth width calculation
        // In production, measure actual landmark distances
        return Math.random(); // Placeholder
    }

    detectFaceCenter(imageData) {
        // Simplified face center detection
        // In production, use face detection API
        return imageData.width / 2 + (Math.random() - 0.5) * 100;
    }

    reset() {
        this.currentChallenge = null;
        this.challengeStartTime = null;
        this.blinkCount = 0;
    }

    isLive(analysisResults) {
        // Determine if user is live based on challenge results
        if (!analysisResults || analysisResults.length === 0) {
            return { isLive: false, confidence: 0, reason: 'No analysis data' };
        }

        const completedChallenges = analysisResults.filter(r => r.completed);
        const avgConfidence = analysisResults.reduce((sum, r) => sum + r.confidence, 0) / analysisResults.length;

        const isLive = completedChallenges.length >= 1 && avgConfidence > 0.6;

        return {
            isLive: isLive,
            confidence: avgConfidence,
            completedChallenges: completedChallenges.length,
            totalChallenges: analysisResults.length,
            reason: isLive ? 'Liveness verified' : 'Failed liveness check'
        };
    }
}

// Export for use in other modules
window.LivenessDetector = LivenessDetector;
