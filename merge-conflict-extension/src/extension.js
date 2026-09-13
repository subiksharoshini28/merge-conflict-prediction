const vscode = require('vscode');
const { GitMonitor } = require('./gitMonitor');
const { FeatureExtractor } = require('./featureExtractor');
const { PredictionClient } = require('./predictionClient');
const { ConflictPanel } = require('./conflictPanel');

let gitMonitor;
let predictionClient;
let statusBarItem;

/**
 * Activate the extension
 */
function activate(context) {
    console.log('Merge Conflict Predictor is now active!');

    // Initialize components
    predictionClient = new PredictionClient();
    gitMonitor = new GitMonitor();
    const featureExtractor = new FeatureExtractor();

    // Create status bar item
    statusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Left,
        100
    );
    statusBarItem.command = 'mergeConflictPredictor.checkCurrent';
    context.subscriptions.push(statusBarItem);

    // Register commands
    registerCommands(context, featureExtractor);

    // Start monitoring
    startMonitoring(context, featureExtractor);

    // Show activation message
    vscode.window.showInformationMessage(
        'Merge Conflict Predictor activated! Use Ctrl+Shift+P to predict conflicts.'
    );
}

/**
 * Register all commands
 */
function registerCommands(context, featureExtractor) {
    // Predict conflict between two branches
    let predictDisposable = vscode.commands.registerCommand(
        'mergeConflictPredictor.predict',
        async () => {
            await predictConflict(context, featureExtractor);
        }
    );
    context.subscriptions.push(predictDisposable);

    // Check current branch
    let checkCurrentDisposable = vscode.commands.registerCommand(
        'mergeConflictPredictor.checkCurrent',
        async () => {
            await checkCurrentBranch(context, featureExtractor);
        }
    );
    context.subscriptions.push(checkCurrentDisposable);

    // Show panel
    let showPanelDisposable = vscode.commands.registerCommand(
        'mergeConflictPredictor.showPanel',
        () => {
            ConflictPanel.createOrShow(context.extensionUri, predictionClient);
        }
    );
    context.subscriptions.push(showPanelDisposable);
}

/**
 * Start monitoring git changes
 */
function startMonitoring(context, featureExtractor) {
    // Check on branch switch
    gitMonitor.onBranchChange(async (branchInfo) => {
        const config = vscode.workspace.getConfiguration('mergeConflictPredictor');
        if (!config.get('autoCheck')) return;

        const risk = await checkConflictRisk(
            featureExtractor,
            branchInfo.branch,
            'main'
        );

        if (risk && risk.probability > config.get('riskThreshold')) {
            showRiskWarning(risk);
        }
    });

    // Start watching
    gitMonitor.start();

    context.subscriptions.push({
        dispose: () => gitMonitor.stop()
    });
}

/**
 * Predict conflict between two branches
 */
async function predictConflict(context, featureExtractor) {
    const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    if (!workspaceRoot) {
        vscode.window.showErrorMessage('No workspace folder open');
        return;
    }

    // Get branch names from user
    const branchA = await vscode.window.showInputBox({
        prompt: 'Enter Branch A name',
        placeHolder: 'feature-login'
    });

    if (!branchA) return;

    const branchB = await vscode.window.showInputBox({
        prompt: 'Enter Branch B name',
        placeHolder: 'feature-signup'
    });

    if (!branchB) return;

    // Show progress
    await vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            title: 'Predicting merge conflict...',
            cancellable: false
        },
        async (progress) => {
            progress.report({ message: 'Extracting features...' });

            try {
                const risk = await checkConflictRisk(
                    featureExtractor,
                    branchA,
                    branchB
                );

                if (risk) {
                    showPredictionResult(risk, branchA, branchB);
                }
            } catch (error) {
                vscode.window.showErrorMessage(
                    `Prediction failed: ${error.message}`
                );
            }
        }
    );
}

/**
 * Check current branch conflict risk
 */
async function checkCurrentBranch(context, featureExtractor) {
    const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    if (!workspaceRoot) {
        vscode.window.showErrorMessage('No workspace folder open');
        return;
    }

    const currentBranch = await gitMonitor.getCurrentBranch(workspaceRoot);
    if (!currentBranch) {
        vscode.window.showErrorMessage('Could not detect current branch');
        return;
    }

    if (currentBranch === 'main' || currentBranch === 'master') {
        vscode.window.showInformationMessage(
            'You are on the main branch. Switch to a feature branch to check risk.'
        );
        return;
    }

    await vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            title: `Checking conflict risk for ${currentBranch}...`,
            cancellable: false
        },
        async (progress) => {
            try {
                const risk = await checkConflictRisk(
                    featureExtractor,
                    currentBranch,
                    'main'
                );

                if (risk) {
                    updateStatusBar(risk);
                    showPredictionResult(risk, currentBranch, 'main');
                }
            } catch (error) {
                vscode.window.showErrorMessage(
                    `Check failed: ${error.message}`
                );
            }
        }
    );
}

/**
 * Check conflict risk between two branches
 */
async function checkConflictRisk(featureExtractor, branchA, branchB) {
    const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;

    // Extract features
    const features = await featureExtractor.extract(
        workspaceRoot,
        branchA,
        branchB
    );

    if (!features) {
        throw new Error('Could not extract features');
    }

    // Get prediction from server
    const prediction = await predictionClient.predict(features);

    return {
        probability: prediction.probability,
        riskLevel: getRiskLevel(prediction.probability),
        features: features,
        branchA: branchA,
        branchB: branchB
    };
}

/**
 * Get risk level from probability
 */
function getRiskLevel(probability) {
    if (probability >= 0.7) return 'HIGH';
    if (probability >= 0.4) return 'MEDIUM';
    return 'LOW';
}

/**
 * Show prediction result
 */
function showPredictionResult(risk, branchA, branchB) {
    const message = `Conflict Risk: ${risk.riskLevel} (${(risk.probability * 100).toFixed(1)}%)`;
    const details = `Between ${branchA} and ${branchB}`;

    if (risk.riskLevel === 'HIGH') {
        vscode.window.showErrorMessage(`${message}\n${details}`, 'Merge Main First');
    } else if (risk.riskLevel === 'MEDIUM') {
        vscode.window.showWarningMessage(`${message}\n${details}`);
    } else {
        vscode.window.showInformationMessage(`${message}\n${details}`);
    }
}

/**
 * Show risk warning
 */
function showRiskWarning(risk) {
    const message = `High conflict risk detected! (${(risk.probability * 100).toFixed(1)}%)`;
    vscode.window.showWarningMessage(message, 'View Details', 'Dismiss');
}

/**
 * Update status bar with risk level
 */
function updateStatusBar(risk) {
    const icon = risk.riskLevel === 'HIGH' ? '$(alert)' :
                 risk.riskLevel === 'MEDIUM' ? '$(warning)' : '$(check)';

    statusBarItem.text = `${icon} Conflict: ${risk.riskLevel}`;
    statusBarItem.tooltip = `Conflict probability: ${(risk.probability * 100).toFixed(1)}%`;
    statusBarItem.color = risk.riskLevel === 'HIGH' ? new vscode.ThemeColor('errorForeground') :
                          risk.riskLevel === 'MEDIUM' ? new vscode.ThemeColor('warningForeground') :
                          new vscode.ThemeColor('successForeground');
    statusBarItem.show();
}

/**
 * Deactivate the extension
 */
function deactivate() {
    if (gitMonitor) {
        gitMonitor.stop();
    }
}

module.exports = {
    activate,
    deactivate
};
