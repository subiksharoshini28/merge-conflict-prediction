const http = require('http');

/**
 * Client for communicating with the prediction server
 */
class PredictionClient {
    constructor(serverUrl = 'http://localhost:5000') {
        this.serverUrl = serverUrl;
    }

    /**
     * Get prediction for features
     */
    async predict(features) {
        return new Promise((resolve, reject) => {
            const postData = JSON.stringify(features);

            const options = {
                hostname: 'localhost',
                port: 5000,
                path: '/predict',
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Content-Length': Buffer.byteLength(postData)
                }
            };

            const req = http.request(options, (res) => {
                let data = '';

                res.on('data', (chunk) => {
                    data += chunk;
                });

                res.on('end', () => {
                    try {
                        const result = JSON.parse(data);
                        resolve(result);
                    } catch (error) {
                        reject(new Error('Invalid response from server'));
                    }
                });
            });

            req.on('error', (error) => {
                // If server is not running, use local prediction
                console.log('Server not available, using local prediction');
                resolve(this.localPredict(features));
            });

            req.write(postData);
            req.end();
        });
    }

    /**
     * Local prediction when server is not available
     * Uses the same logic as the trained model
     */
    localPredict(features) {
        // Simple rule-based prediction based on features
        let score = 0;

        // Overlap is the strongest predictor
        if (features.overlap_files > 0) {
            score += features.overlap_files * 0.3;
        }

        // High churn increases risk
        if (features.churn_p1 > 100) score += 0.1;
        if (features.churn_p2 > 100) score += 0.1;

        // Many commits increase risk
        if (features.commits_p1 > 10) score += 0.05;
        if (features.commits_p2 > 10) score += 0.05;

        // Many authors increase risk
        if (features.authors_p1 > 3) score += 0.05;
        if (features.authors_p2 > 3) score += 0.05;

        // Overlap ratio
        score += features.overlap_ratio * 0.2;

        // Cap at 1.0
        const probability = Math.min(score, 1.0);

        return {
            probability: probability,
            risk_level: this.getRiskLevel(probability),
            features: features
        };
    }

    /**
     * Get risk level from probability
     */
    getRiskLevel(probability) {
        if (probability >= 0.7) return 'HIGH';
        if (probability >= 0.4) return 'MEDIUM';
        return 'LOW';
    }

    /**
     * Check if server is running
     */
    async checkServer() {
        return new Promise((resolve) => {
            const req = http.request(
                {
                    hostname: 'localhost',
                    port: 5000,
                    path: '/health',
                    method: 'GET',
                    timeout: 2000
                },
                (res) => {
                    resolve(res.statusCode === 200);
                }
            );

            req.on('error', () => resolve(false));
            req.on('timeout', () => {
                req.destroy();
                resolve(false);
            });

            req.end();
        });
    }
}

module.exports = { PredictionClient };
