const vscode = require('vscode');

/**
 * Webview panel for conflict prediction
 */
class ConflictPanel {
    static currentPanel = undefined;
    static viewType = 'mergeConflictPredictor.panel';

    constructor(extensionUri, predictionClient) {
        this._extensionUri = extensionUri;
        this._predictionClient = predictionClient;
        this._disposables = [];
    }

    /**
     * Create or show the panel
     */
    static createOrShow(extensionUri, predictionClient) {
        const column = vscode.ViewColumn.One;

        if (ConflictPanel.currentPanel) {
            ConflictPanel.currentPanel._panel.reveal(column);
            return;
        }

        const panel = vscode.window.createWebviewPanel(
            ConflictPanel.viewType,
            'Conflict Predictor',
            column,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );

        ConflictPanel.currentPanel = new ConflictPanel(panel, extensionUri, predictionClient);
    }

    /**
     * Constructor
     */
    constructor(panel, extensionUri, predictionClient) {
        this._panel = panel;
        this._extensionUri = extensionUri;
        this._predictionClient = predictionClient;

        this._panel.webview.html = this._getHtmlForWebview();

        this._panel.onDidDispose(() => this.dispose(), null, this._disposables);
    }

    /**
     * Get HTML for webview
     */
    _getHtmlForWebview() {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Merge Conflict Predictor</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            color: var(--vscode-foreground);
            background: var(--vscode-editor-background);
            padding: 20px;
        }
        h1 { color: var(--vscode-titleBar-activeForeground); }
        .input-group { margin: 15px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select {
            width: 100%;
            padding: 8px;
            border: 1px solid var(--vscode-input-border);
            background: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border-radius: 4px;
        }
        button {
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            margin: 5px;
        }
        button:hover { background: var(--vscode-button-hoverBackground); }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            display: none;
        }
        .risk-low { background: #1a472a; border: 1px solid #2ecc71; }
        .risk-medium { background: #4a3000; border: 1px solid #f39c12; }
        .risk-high { background: #4a1a1a; border: 1px solid #e74c3c; }
        .probability {
            font-size: 48px;
            font-weight: bold;
            text-align: center;
        }
        .risk-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            text-align: center;
        }
        .features {
            margin-top: 15px;
            padding: 10px;
            background: var(--vscode-editor-background);
            border-radius: 4px;
        }
        .feature-row {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid var(--vscode-panel-border);
        }
    </style>
</head>
<body>
    <h1>Merge Conflict Predictor</h1>
    
    <div class="input-group">
        <label>Repository (local path or GitHub URL)</label>
        <input type="text" id="repo" placeholder="/path/to/repo or https://github.com/user/repo">
    </div>
    
    <div class="input-group">
        <label>Branch A</label>
        <input type="text" id="branchA" placeholder="feature-login">
    </div>
    
    <div class="input-group">
        <label>Branch B</label>
        <input type="text" id="branchB" placeholder="feature-signup">
    </div>
    
    <button onclick="predict()">Predict Conflict</button>
    <button onclick="useCurrentBranch()">Use Current Branch</button>
    
    <div id="result" class="result">
        <div class="probability" id="probability">0%</div>
        <div class="risk-badge" id="riskBadge">LOW RISK</div>
        <div class="features" id="features"></div>
    </div>

    <script>
        const vscode = acquireVsCodeApi();

        function predict() {
            const repo = document.getElementById('repo').value;
            const branchA = document.getElementById('branchA').value;
            const branchB = document.getElementById('branchB').value;

            if (!repo || !branchA || !branchB) {
                alert('Please fill in all fields');
                return;
            }

            // Send message to extension
            vscode.postMessage({
                type: 'predict',
                repo: repo,
                branchA: branchA,
                branchB: branchB
            });
        }

        function useCurrentBranch() {
            vscode.postMessage({ type: 'getCurrentBranch' });
        }

        // Handle messages from extension
        window.addEventListener('message', (event) => {
            const message = event.data;
            
            if (message.type === 'predictionResult') {
                showResult(message.result);
            }
        });

        function showResult(result) {
            const resultDiv = document.getElementById('result');
            const probabilityDiv = document.getElementById('probability');
            const riskBadge = document.getElementById('riskBadge');
            const featuresDiv = document.getElementById('features');

            const probability = (result.probability * 100).toFixed(1);
            probabilityDiv.textContent = probability + '%';

            let riskClass = 'risk-low';
            let riskText = 'LOW RISK';
            
            if (result.probability >= 0.7) {
                riskClass = 'risk-high';
                riskText = 'HIGH RISK';
            } else if (result.probability >= 0.4) {
                riskClass = 'risk-medium';
                riskText = 'MEDIUM RISK';
            }

            riskBadge.textContent = riskText;
            riskBadge.className = 'risk-badge ' + riskClass;
            resultDiv.className = 'result ' + riskClass;

            // Show features
            if (result.features) {
                featuresDiv.innerHTML = '<h3>Features:</h3>' +
                    '<div class="feature-row"><span>Overlap Files</span><span>' + result.features.overlap_files + '</span></div>' +
                    '<div class="feature-row"><span>Overlap Ratio</span><span>' + result.features.overlap_ratio.toFixed(2) + '</span></div>' +
                    '<div class="feature-row"><span>Churn P1</span><span>' + result.features.churn_p1 + '</span></div>' +
                    '<div class="feature-row"><span>Churn P2</span><span>' + result.features.churn_p2 + '</span></div>';
            }

            resultDiv.style.display = 'block';
        }
    </script>
</body>
</html>`;
    }

    /**
     * Dispose the panel
     */
    dispose() {
        ConflictPanel.currentPanel = undefined;
        this._panel.dispose();
        while (this._disposables.length) {
            const disposable = this._disposables.pop();
            disposable.dispose();
        }
    }
}

module.exports = { ConflictPanel };
