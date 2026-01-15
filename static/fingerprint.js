// Device Fingerprinting Module
// Collects unique device characteristics for identification

class DeviceFingerprint {
    constructor() {
        this.fingerprint = null;
    }

    async generate() {
        const components = {
            // Browser & OS
            userAgent: navigator.userAgent,
            platform: navigator.platform,
            language: navigator.language,
            languages: navigator.languages?.join(',') || '',

            // Screen
            screenResolution: `${screen.width}x${screen.height}`,
            screenDepth: screen.colorDepth,
            screenAvailSize: `${screen.availWidth}x${screen.availHeight}`,

            // Timezone
            timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
            timezoneOffset: new Date().getTimezoneOffset(),

            // Hardware
            hardwareConcurrency: navigator.hardwareConcurrency || 0,
            deviceMemory: navigator.deviceMemory || 0,
            maxTouchPoints: navigator.maxTouchPoints || 0,

            // Browser Features
            cookieEnabled: navigator.cookieEnabled,
            doNotTrack: navigator.doNotTrack || 'unknown',

            // Fonts (basic detection)
            fonts: await this.detectFonts(),

            // Canvas Fingerprint
            canvas: await this.getCanvasFingerprint(),

            // WebGL Fingerprint
            webgl: await this.getWebGLFingerprint(),

            // Audio Context
            audio: await this.getAudioFingerprint()
        };

        // Generate hash from components
        this.fingerprint = await this.hashComponents(components);
        return {
            fingerprint: this.fingerprint,
            components: components
        };
    }

    async detectFonts() {
        const baseFonts = ['monospace', 'sans-serif', 'serif'];
        const testFonts = [
            'Arial', 'Verdana', 'Times New Roman', 'Courier New',
            'Georgia', 'Palatino', 'Garamond', 'Comic Sans MS',
            'Trebuchet MS', 'Impact'
        ];

        const detected = [];
        const testString = 'mmmmmmmmmmlli';
        const testSize = '72px';

        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');

        for (const font of testFonts) {
            let detected_font = false;
            for (const baseFont of baseFonts) {
                context.font = `${testSize} ${baseFont}`;
                const baseWidth = context.measureText(testString).width;

                context.font = `${testSize} ${font}, ${baseFont}`;
                const testWidth = context.measureText(testString).width;

                if (baseWidth !== testWidth) {
                    detected_font = true;
                    break;
                }
            }
            if (detected_font) detected.push(font);
        }

        return detected.join(',');
    }

    async getCanvasFingerprint() {
        try {
            const canvas = document.createElement('canvas');
            canvas.width = 200;
            canvas.height = 50;
            const ctx = canvas.getContext('2d');

            // Draw text with various styles
            ctx.textBaseline = 'top';
            ctx.font = '14px "Arial"';
            ctx.textBaseline = 'alphabetic';
            ctx.fillStyle = '#f60';
            ctx.fillRect(125, 1, 62, 20);
            ctx.fillStyle = '#069';
            ctx.fillText('BroCode Sentinel 🔒', 2, 15);
            ctx.fillStyle = 'rgba(102, 204, 0, 0.7)';
            ctx.fillText('Device Fingerprint', 4, 17);

            return canvas.toDataURL();
        } catch (e) {
            return 'canvas_error';
        }
    }

    async getWebGLFingerprint() {
        try {
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');

            if (!gl) return 'no_webgl';

            const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
            const vendor = debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : 'unknown';
            const renderer = debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : 'unknown';

            return `${vendor}|${renderer}`;
        } catch (e) {
            return 'webgl_error';
        }
    }

    async getAudioFingerprint() {
        try {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            if (!AudioContext) return 'no_audio';

            const context = new AudioContext();
            const oscillator = context.createOscillator();
            const analyser = context.createAnalyser();
            const gainNode = context.createGain();
            const scriptProcessor = context.createScriptProcessor(4096, 1, 1);

            gainNode.gain.value = 0; // Mute
            oscillator.connect(analyser);
            analyser.connect(scriptProcessor);
            scriptProcessor.connect(gainNode);
            gainNode.connect(context.destination);

            oscillator.start(0);

            return new Promise((resolve) => {
                scriptProcessor.onaudioprocess = function (event) {
                    const output = event.outputBuffer.getChannelData(0);
                    const sum = output.reduce((a, b) => a + Math.abs(b), 0);
                    oscillator.stop();
                    scriptProcessor.disconnect();
                    context.close();
                    resolve(sum.toString());
                };
            });
        } catch (e) {
            return 'audio_error';
        }
    }

    async hashComponents(components) {
        const str = JSON.stringify(components);
        const encoder = new TextEncoder();
        const data = encoder.encode(str);
        const hashBuffer = await crypto.subtle.digest('SHA-256', data);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    }

    getFingerprint() {
        return this.fingerprint;
    }
}

// Export for use in other modules
window.DeviceFingerprint = DeviceFingerprint;
