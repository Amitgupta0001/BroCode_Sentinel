// Explainable Trust UI Module
// Provides transparency into trust score calculations

class ExplainableTrustUI {
    constructor() {
        this.container = null;
        this.chartInstance = null;
    }

    /**
     * Initialize explainable trust UI
     */
    init(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error('Explainable Trust UI container not found');
            return;
        }

        this.render();
    }

    /**
     * Render the explainable trust UI
     */
    render() {
        this.container.innerHTML = `
      <div class="explainable-trust-panel">
        <div class="trust-header">
          <h3>🔍 Trust Score Breakdown</h3>
          <p class="subtitle">Understanding your security score</p>
        </div>

        <div class="trust-main-score">
          <div class="score-circle" id="main-score-circle">
            <div class="score-value" id="main-score-value">--</div>
            <div class="score-label">Overall Trust</div>
          </div>
        </div>

        <div class="trust-components" id="trust-components">
          <!-- Components will be inserted here -->
        </div>

        <div class="trust-factors" id="trust-factors">
          <h4>📊 Contributing Factors</h4>
          <div class="factors-list" id="factors-list">
            <!-- Factors will be inserted here -->
          </div>
        </div>

        <div class="trust-trend">
          <h4>📈 Trust Score Trend</h4>
          <canvas id="trust-trend-chart"></canvas>
        </div>

        <div class="trust-recommendations" id="trust-recommendations">
          <h4>💡 Recommendations</h4>
          <div class="recommendations-list" id="recommendations-list">
            <!-- Recommendations will be inserted here -->
          </div>
        </div>
      </div>
    `;

        this.initTrendChart();
    }

    /**
     * Update trust score display with detailed breakdown
     */
    updateTrustScore(trustData) {
        // Update main score
        const mainScore = Math.round(trustData.fused_score * 100);
        const scoreValue = document.getElementById('main-score-value');
        const scoreCircle = document.getElementById('main-score-circle');

        if (scoreValue) {
            scoreValue.textContent = `${mainScore}%`;

            // Color code based on score
            scoreCircle.className = 'score-circle';
            if (mainScore >= 80) {
                scoreCircle.classList.add('excellent');
            } else if (mainScore >= 60) {
                scoreCircle.classList.add('good');
            } else if (mainScore >= 40) {
                scoreCircle.classList.add('warning');
            } else {
                scoreCircle.classList.add('critical');
            }
        }

        // Update components breakdown
        this.updateComponents(trustData.contributions);

        // Update factors
        this.updateFactors(trustData);

        // Update recommendations
        this.updateRecommendations(trustData);

        // Update trend chart
        this.updateTrendChart(trustData.fused_score);
    }

    /**
     * Update component scores with visual bars
     */
    updateComponents(contributions) {
        const container = document.getElementById('trust-components');
        if (!container) return;

        const components = [
            { name: 'Keystroke Dynamics', key: 'keystroke', icon: '⌨️' },
            { name: 'Facial Recognition', key: 'face', icon: '👤' },
            { name: 'Behavioral Patterns', key: 'behavior', icon: '🧠' },
            { name: 'Liveness Detection', key: 'liveness', icon: '✨' },
            { name: 'Voice Biometrics', key: 'voice', icon: '🎤' }
        ];

        let html = '<div class="components-grid">';

        components.forEach(component => {
            const data = contributions[component.key] || { score: 0, weight: 0, contribution: 0 };
            const score = Math.round(data.score * 100);
            const weight = Math.round(data.weight * 100);
            const contribution = Math.round(data.contribution * 100);

            html += `
        <div class="component-card">
          <div class="component-header">
            <span class="component-icon">${component.icon}</span>
            <span class="component-name">${component.name}</span>
          </div>
          <div class="component-score">
            <div class="score-bar-container">
              <div class="score-bar" style="width: ${score}%; background: ${this.getScoreColor(score)}"></div>
            </div>
            <span class="score-text">${score}%</span>
          </div>
          <div class="component-details">
            <span class="detail-item">Weight: ${weight}%</span>
            <span class="detail-item">Impact: ${contribution}%</span>
          </div>
        </div>
      `;
        });

        html += '</div>';
        container.innerHTML = html;
    }

    /**
     * Update contributing factors
     */
    updateFactors(trustData) {
        const container = document.getElementById('factors-list');
        if (!container) return;

        const factors = this.analyzeFactors(trustData);

        let html = '';
        factors.forEach(factor => {
            const icon = factor.impact === 'positive' ? '✅' :
                factor.impact === 'negative' ? '⚠️' : 'ℹ️';

            html += `
        <div class="factor-item ${factor.impact}">
          <span class="factor-icon">${icon}</span>
          <span class="factor-text">${factor.text}</span>
          <span class="factor-value">${factor.value}</span>
        </div>
      `;
        });

        container.innerHTML = html || '<p class="no-factors">No significant factors detected</p>';
    }

    /**
     * Analyze factors contributing to trust score
     */
    analyzeFactors(trustData) {
        const factors = [];
        const contributions = trustData.contributions || {};

        // Analyze each component
        Object.entries(contributions).forEach(([key, data]) => {
            const score = data.score || 0;
            const weight = data.weight || 0;

            if (score >= 0.8 && weight > 0.1) {
                factors.push({
                    text: `${key.charAt(0).toUpperCase() + key.slice(1)} performing well`,
                    value: `${Math.round(score * 100)}%`,
                    impact: 'positive'
                });
            } else if (score < 0.5 && weight > 0.1) {
                factors.push({
                    text: `${key.charAt(0).toUpperCase() + key.slice(1)} needs attention`,
                    value: `${Math.round(score * 100)}%`,
                    impact: 'negative'
                });
            }
        });

        // Analyze adaptive weights
        if (trustData.adaptive_weights) {
            const weights = trustData.adaptive_weights;
            const maxWeight = Math.max(...Object.values(weights));
            const maxKey = Object.keys(weights).find(k => weights[k] === maxWeight);

            if (maxKey) {
                factors.push({
                    text: `Primary authentication method`,
                    value: maxKey.charAt(0).toUpperCase() + maxKey.slice(1),
                    impact: 'neutral'
                });
            }
        }

        // Analyze smoothing
        if (trustData.smoothed_score !== undefined && trustData.raw_score !== undefined) {
            const diff = Math.abs(trustData.smoothed_score - trustData.raw_score);
            if (diff > 0.1) {
                factors.push({
                    text: 'Temporal smoothing applied',
                    value: `±${Math.round(diff * 100)}%`,
                    impact: 'neutral'
                });
            }
        }

        return factors;
    }

    /**
     * Update recommendations
     */
    updateRecommendations(trustData) {
        const container = document.getElementById('recommendations-list');
        if (!container) return;

        const recommendations = this.generateRecommendations(trustData);

        let html = '';
        recommendations.forEach(rec => {
            html += `
        <div class="recommendation-item ${rec.priority}">
          <span class="rec-icon">${rec.icon}</span>
          <div class="rec-content">
            <div class="rec-title">${rec.title}</div>
            <div class="rec-description">${rec.description}</div>
          </div>
        </div>
      `;
        });

        container.innerHTML = html || '<p class="no-recommendations">✅ Everything looks good!</p>';
    }

    /**
     * Generate recommendations based on trust data
     */
    generateRecommendations(trustData) {
        const recommendations = [];
        const contributions = trustData.contributions || {};

        // Check each component
        Object.entries(contributions).forEach(([key, data]) => {
            const score = data.score || 0;

            if (score < 0.5) {
                let rec = null;

                switch (key) {
                    case 'keystroke':
                        rec = {
                            icon: '⌨️',
                            title: 'Improve Typing Consistency',
                            description: 'Your typing pattern differs from normal. Type naturally and avoid rushing.',
                            priority: 'medium'
                        };
                        break;
                    case 'face':
                        rec = {
                            icon: '👤',
                            title: 'Improve Camera Position',
                            description: 'Ensure your face is clearly visible and well-lit.',
                            priority: 'high'
                        };
                        break;
                    case 'behavior':
                        rec = {
                            icon: '🧠',
                            title: 'Behavioral Pattern Alert',
                            description: 'Your activity pattern is unusual. Continue normal work to rebuild trust.',
                            priority: 'medium'
                        };
                        break;
                    case 'liveness':
                        rec = {
                            icon: '✨',
                            title: 'Liveness Check Failed',
                            description: 'Complete a liveness challenge to verify your presence.',
                            priority: 'high'
                        };
                        break;
                    case 'voice':
                        rec = {
                            icon: '🎤',
                            title: 'Voice Verification Needed',
                            description: 'Your voice pattern is unclear. Speak clearly for verification.',
                            priority: 'medium'
                        };
                        break;
                }

                if (rec) recommendations.push(rec);
            }
        });

        // Overall trust recommendations
        const overallScore = trustData.fused_score || 0;
        if (overallScore < 0.4) {
            recommendations.unshift({
                icon: '🚨',
                title: 'Critical: Trust Score Low',
                description: 'Your trust score is critically low. Complete re-authentication to continue.',
                priority: 'critical'
            });
        } else if (overallScore < 0.6) {
            recommendations.unshift({
                icon: '⚠️',
                title: 'Warning: Trust Score Declining',
                description: 'Your trust score is below normal. Follow recommendations to improve.',
                priority: 'high'
            });
        }

        return recommendations;
    }

    /**
     * Initialize trend chart
     */
    initTrendChart() {
        const canvas = document.getElementById('trust-trend-chart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        this.chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Trust Score',
                    data: [],
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: (value) => value + '%'
                        }
                    }
                }
            }
        });
    }

    /**
     * Update trend chart with new data
     */
    updateTrendChart(score) {
        if (!this.chartInstance) return;

        const timestamp = new Date().toLocaleTimeString();
        const scorePercent = Math.round(score * 100);

        // Add new data point
        this.chartInstance.data.labels.push(timestamp);
        this.chartInstance.data.datasets[0].data.push(scorePercent);

        // Keep only last 20 points
        if (this.chartInstance.data.labels.length > 20) {
            this.chartInstance.data.labels.shift();
            this.chartInstance.data.datasets[0].data.shift();
        }

        this.chartInstance.update();
    }

    /**
     * Get color based on score
     */
    getScoreColor(score) {
        if (score >= 80) return '#10b981';
        if (score >= 60) return '#3b82f6';
        if (score >= 40) return '#f59e0b';
        return '#ef4444';
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ExplainableTrustUI;
}
