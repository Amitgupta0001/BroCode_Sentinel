// Voice Biometrics Module
// Implements speaker recognition for authentication

class VoiceBiometrics {
    constructor() {
        this.audioContext = null;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.stream = null;
        this.isRecording = false;
        this.sampleRate = 16000; // Standard for speech recognition
    }

    async initialize() {
        try {
            // Request microphone access
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: this.sampleRate
                }
            });

            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.sampleRate
            });

            console.log('Voice biometrics initialized');
            return true;
        } catch (err) {
            console.error('Microphone access denied:', err);
            return false;
        }
    }

    async startRecording(duration = 3000) {
        if (!this.stream) {
            throw new Error('Voice biometrics not initialized');
        }

        this.audioChunks = [];
        this.isRecording = true;

        // Create media recorder
        this.mediaRecorder = new MediaRecorder(this.stream, {
            mimeType: 'audio/webm;codecs=opus'
        });

        // Collect audio data
        this.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                this.audioChunks.push(event.data);
            }
        };

        // Start recording
        this.mediaRecorder.start();
        console.log('Recording started...');

        // Auto-stop after duration
        return new Promise((resolve) => {
            setTimeout(async () => {
                await this.stopRecording();
                resolve(this.audioChunks);
            }, duration);
        });
    }

    async stopRecording() {
        if (!this.mediaRecorder || !this.isRecording) {
            return null;
        }

        return new Promise((resolve) => {
            this.mediaRecorder.onstop = async () => {
                this.isRecording = false;

                // Create audio blob
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });

                // Convert to audio buffer
                const arrayBuffer = await audioBlob.arrayBuffer();
                const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);

                console.log('Recording stopped. Duration:', audioBuffer.duration, 's');
                resolve({
                    blob: audioBlob,
                    buffer: audioBuffer,
                    duration: audioBuffer.duration
                });
            };

            this.mediaRecorder.stop();
        });
    }

    extractMFCC(audioBuffer, numCoefficients = 13) {
        /**
         * Extract Mel-Frequency Cepstral Coefficients (MFCC)
         * These are the standard features for speaker recognition
         */

        const channelData = audioBuffer.getChannelData(0);
        const frameSize = 512;
        const hopSize = 256;
        const numFrames = Math.floor((channelData.length - frameSize) / hopSize);

        const mfccFeatures = [];

        for (let i = 0; i < numFrames; i++) {
            const start = i * hopSize;
            const frame = channelData.slice(start, start + frameSize);

            // Apply Hamming window
            const windowedFrame = this.applyHammingWindow(frame);

            // Compute FFT
            const spectrum = this.computeFFT(windowedFrame);

            // Apply Mel filterbank
            const melSpectrum = this.applyMelFilterbank(spectrum, numCoefficients);

            // Compute DCT (Discrete Cosine Transform)
            const mfcc = this.computeDCT(melSpectrum, numCoefficients);

            mfccFeatures.push(mfcc);
        }

        // Compute statistics over all frames
        const mfccStats = this.computeStatistics(mfccFeatures);

        return mfccStats;
    }

    applyHammingWindow(frame) {
        const N = frame.length;
        const windowed = new Float32Array(N);

        for (let n = 0; n < N; n++) {
            const window = 0.54 - 0.46 * Math.cos(2 * Math.PI * n / (N - 1));
            windowed[n] = frame[n] * window;
        }

        return windowed;
    }

    computeFFT(frame) {
        // Simplified FFT (in production, use a library like fft.js)
        const N = frame.length;
        const spectrum = new Float32Array(N / 2);

        for (let k = 0; k < N / 2; k++) {
            let real = 0;
            let imag = 0;

            for (let n = 0; n < N; n++) {
                const angle = -2 * Math.PI * k * n / N;
                real += frame[n] * Math.cos(angle);
                imag += frame[n] * Math.sin(angle);
            }

            spectrum[k] = Math.sqrt(real * real + imag * imag);
        }

        return spectrum;
    }

    applyMelFilterbank(spectrum, numFilters) {
        // Simplified Mel filterbank
        const melSpectrum = new Float32Array(numFilters);
        const spectrumLength = spectrum.length;
        const filterWidth = Math.floor(spectrumLength / numFilters);

        for (let i = 0; i < numFilters; i++) {
            const start = i * filterWidth;
            const end = Math.min(start + filterWidth, spectrumLength);

            let sum = 0;
            for (let j = start; j < end; j++) {
                sum += spectrum[j];
            }

            melSpectrum[i] = sum / filterWidth;
        }

        return melSpectrum;
    }

    computeDCT(melSpectrum, numCoefficients) {
        // Discrete Cosine Transform
        const N = melSpectrum.length;
        const dct = new Float32Array(numCoefficients);

        for (let k = 0; k < numCoefficients; k++) {
            let sum = 0;

            for (let n = 0; n < N; n++) {
                sum += melSpectrum[n] * Math.cos(Math.PI * k * (n + 0.5) / N);
            }

            dct[k] = sum;
        }

        return dct;
    }

    computeStatistics(features) {
        // Compute mean and variance for each coefficient
        const numCoefficients = features[0].length;
        const numFrames = features.length;

        const mean = new Float32Array(numCoefficients);
        const variance = new Float32Array(numCoefficients);

        // Compute mean
        for (let i = 0; i < numCoefficients; i++) {
            let sum = 0;
            for (let j = 0; j < numFrames; j++) {
                sum += features[j][i];
            }
            mean[i] = sum / numFrames;
        }

        // Compute variance
        for (let i = 0; i < numCoefficients; i++) {
            let sum = 0;
            for (let j = 0; j < numFrames; j++) {
                const diff = features[j][i] - mean[i];
                sum += diff * diff;
            }
            variance[i] = sum / numFrames;
        }

        return {
            mean: Array.from(mean),
            variance: Array.from(variance),
            numFrames: numFrames
        };
    }

    computeSimilarity(features1, features2) {
        /**
         * Compute cosine similarity between two feature vectors
         * Returns value between 0 (different) and 1 (identical)
         */

        const mean1 = features1.mean;
        const mean2 = features2.mean;

        let dotProduct = 0;
        let norm1 = 0;
        let norm2 = 0;

        for (let i = 0; i < mean1.length; i++) {
            dotProduct += mean1[i] * mean2[i];
            norm1 += mean1[i] * mean1[i];
            norm2 += mean2[i] * mean2[i];
        }

        const similarity = dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));

        // Normalize to 0-1 range
        return (similarity + 1) / 2;
    }

    async recordAndExtractFeatures(duration = 3000) {
        /**
         * Complete workflow: record audio and extract MFCC features
         */

        const recording = await this.startRecording(duration);
        const features = this.extractMFCC(recording.buffer);

        return {
            features: features,
            duration: recording.duration,
            blob: recording.blob
        };
    }

    cleanup() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }
        if (this.audioContext) {
            this.audioContext.close();
        }
    }
}

// Export for use in other modules
window.VoiceBiometrics = VoiceBiometrics;
