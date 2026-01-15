// Explainable AI Module
// Provides transparency into trust score calculations

class ExplainableAI {
    constructor() {
        this.scoreHistory = [];
        this.maxHistory = 50;
    }

    explainTrustScore(scoreData) {
        /**
         * Break down trust score into understandable components
         * 
         * scoreData: {
         *   final_trust: 0.75,
         *   components: {
         *     keystroke: 0.8,
         *     face: 0.7,
         *     behavior: 0.75,
         *     liveness: 0.9,
         *     voice: 0.85
         *   },
         *   weights: {
         *     keystroke: 0.2,
         *     face: 0.3,
         *     behavior: 0.2,
         *     liveness: 0.15,
         *     voice: 0.15
         *   }
         * }
         */

        const components = scoreData.components || {};
        const weights = scoreData.weights || {};
        const finalTrust = scoreData.final_trust || 0;

        // Calculate contributions
        const contributions = {};
        let totalContribution = 0;

        for (const [key, score] of Object.entries(components)) {
            const weight = weights[key] || 0;
            const contribution = score * weight;
            contributions[key] = {
                score: score,
                weight: weight,
                contribution: contribution,
                percentage: 0 // Will calculate after total
            };
            totalContribution += contribution;
        }

        // Calculate percentages
        for (const key in contributions) {
            contributions[key].percentage = totalContribution > 0
                ? (contributions[key].contribution / totalContribution) * 100
                : 0;
        }

        // Identify top positive and negative factors
        const sortedFactors = Object.entries(contributions)
            .sort((a, b) => b[1].contribution - a[1].contribution);

        const topPositive = sortedFactors.slice(0, 2);
        const topNegative = sortedFactors.slice(-2).reverse();

        // Generate human-readable explanation
        const explanation = this.generateExplanation(
            finalTrust,
            contributions,
            topPositive,
            topNegative
        );

        // Store in history
        this.scoreHistory.push({
            timestamp: Date.now(),
            trust: finalTrust,
            components: components,
            explanation: explanation
        });

        if (this.scoreHistory.length > this.maxHistory) {
            this.scoreHistory.shift();
        }

        return {
            trust: finalTrust,
            contributions: contributions,
            topPositive: topPositive,
            topNegative: topNegative,
            explanation: explanation,
            trend: this.calculateTrend()
        };
    }

    generateExplanation(trust, contributions, topPositive, topNegative) {
        const messages = [];

        // Overall status
        if (trust >= 0.8) {
            messages.push("✅ Excellent - All security checks passed");
        } else if (trust >= 0.5) {
            messages.push("⚠️ Good - Minor security concerns detected");
        } else if (trust >= 0.3) {
            messages.push("⚠️ Warning - Significant security concerns");
        } else {
            messages.push("🚨 Critical - Multiple security failures");
        }

        // Top positive factors
        if (topPositive.length > 0) {
            const [factor, data] = topPositive[0];
            messages.push(`👍 Strongest: ${this.formatFactorName(factor)} (${(data.score * 100).toFixed(0)}%)`);
        }

        // Top negative factors
        if (topNegative.length > 0) {
            const [factor, data] = topNegative[0];
            if (data.score < 0.5) {
                messages.push(`👎 Weakest: ${this.formatFactorName(factor)} (${(data.score * 100).toFixed(0)}%)`);
            }
        }

        // Specific recommendations
        const recommendations = this.generateRecommendations(contributions);
        if (recommendations.length > 0) {
            messages.push(...recommendations);
        }

        return messages;
    }

    generateRecommendations(contributions) {
        const recommendations = [];

        for (const [factor, data] of Object.entries(contributions)) {
            if (data.score < 0.5) {
                switch (factor) {
                    case 'face':
                        recommendations.push("💡 Ensure your face is clearly visible to the camera");
                        break;
                    case 'keystroke':
                        recommendations.push("💡 Type naturally at your normal speed");
                        break;
                    case 'liveness':
                        recommendations.push("💡 Complete liveness challenges when prompted");
                        break;
                    case 'voice':
                        recommendations.push("💡 Speak clearly when using voice authentication");
                        break;
                    case 'behavior':
                        recommendations.push("💡 Unusual behavior detected - this may be normal if you're in a new environment");
                        break;
                }
            }
        }

        return recommendations.slice(0, 2); // Max 2 recommendations
    }

    formatFactorName(factor) {
        const names = {
            'keystroke': 'Typing Pattern',
            'face': 'Face Recognition',
            'behavior': 'Behavior Pattern',
            'liveness': 'Liveness Check',
            'voice': 'Voice Recognition',
            'device': 'Device Fingerprint',
            'location': 'Location Verification'
        };
        return names[factor] || factor;
    }

    calculateTrend() {
        if (this.scoreHistory.length < 3) {
            return 'stable';
        }

        const recent = this.scoreHistory.slice(-5);
        const scores = recent.map(h => h.trust);

        // Calculate simple linear trend
        let increasing = 0;
        let decreasing = 0;

        for (let i = 1; i < scores.length; i++) {
            if (scores[i] > scores[i - 1]) increasing++;
            if (scores[i] < scores[i - 1]) decreasing++;
        }

        if (increasing > decreasing + 1) return 'improving';
        if (decreasing > increasing + 1) return 'declining';
        return 'stable';
    }

    getScoreHistory(limit = 20) {
        return this.scoreHistory.slice(-limit);
    }

    getFactorImportance() {
        /**
         * Calculate which factors have the most impact on trust score
         */
        if (this.scoreHistory.length < 5) {
            return {};
        }

        const recent = this.scoreHistory.slice(-10);
        const factorImpact = {};

        for (const record of recent) {
            for (const [factor, score] of Object.entries(record.components)) {
                if (!factorImpact[factor]) {
                    factorImpact[factor] = {
                        avgScore: 0,
                        variance: 0,
                        count: 0,
                        scores: []
                    };
                }
                factorImpact[factor].scores.push(score);
                factorImpact[factor].count++;
            }
        }

        // Calculate statistics
        for (const factor in factorImpact) {
            const scores = factorImpact[factor].scores;
            const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
            const variance = scores.reduce((sum, s) => sum + Math.pow(s - avg, 2), 0) / scores.length;

            factorImpact[factor].avgScore = avg;
            factorImpact[factor].variance = variance;
            factorImpact[factor].importance = variance; // High variance = high importance
        }

        return factorImpact;
    }

    visualizeContributions(contributions) {
        /**
         * Generate data for visualization
         */
        const data = [];

        for (const [factor, info] of Object.entries(contributions)) {
            data.push({
                name: this.formatFactorName(factor),
                score: Math.round(info.score * 100),
                weight: Math.round(info.weight * 100),
                contribution: Math.round(info.contribution * 100),
                percentage: Math.round(info.percentage)
            });
        }

        // Sort by contribution
        data.sort((a, b) => b.contribution - a.contribution);

        return data;
    }
}

// Export for use in other modules
window.ExplainableAI = ExplainableAI;
