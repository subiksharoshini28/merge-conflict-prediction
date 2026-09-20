# Merge Conflict Predictor - VS Code Extension

AI-powered merge conflict prediction for Git repositories.

## Features

- **Real-time monitoring** - Detects branch switches and warns of conflict risk
- **Conflict prediction** - Uses trained ML model to predict merge conflicts
- **One-click prediction** - Right-click to predict conflicts between branches
- **Visual dashboard** - Webview panel with conflict probability and risk level
- **Auto-check** - Automatically checks conflict risk on branch switch

## Installation

### From VSIX

```bash
cd merge-conflict-extension
npm install
npm run package
code --install-extension merge-conflict-predictor-1.0.0.vsix
```

### From Source

1. Open VS Code
2. Press `Ctrl+Shift+P` -> "Extensions: Install from VSIX..."
3. Select the `.vsix` file

## Usage

### 1. Start the Prediction Server

```bash
cd Software_metrics
python prediction_server.py
```

### 2. Use the Extension

#### Command Palette
- `Ctrl+Shift+P` -> "Predict Merge Conflict"
- `Ctrl+Shift+P` -> "Check Current Branch Conflict Risk"
- `Ctrl+Shift+P` -> "Show Conflict Predictor Panel"

#### Activity Bar
Click the shield icon in the activity bar to open the Conflict Predictor panel.

#### Status Bar
The status bar shows the current branch's conflict risk level.

### 3. Predict Conflicts

1. Open a Git repository in VS Code
2. Switch to a feature branch
3. The extension automatically checks conflict risk with main
4. Or use `Ctrl+Shift+P` -> "Predict Merge Conflict"
5. Enter branch names to compare

## How It Works

### Feature Extraction

The extension extracts 10 git-history features:

| Feature | Description |
|---------|-------------|
| commits_p1 | Number of commits in branch A |
| commits_p2 | Number of commits in branch B |
| files_p1 | Files changed in branch A |
| files_p2 | Files changed in branch B |
| overlap_files | Files changed by both branches |
| overlap_ratio | Overlap as fraction of all files |
| authors_p1 | Distinct authors in branch A |
| authors_p2 | Distinct authors in branch B |
| churn_p1 | Lines changed in branch A |
| churn_p2 | Lines changed in branch B |

### Prediction

The extension uses a trained RandomForest model to predict conflict probability.

- **HIGH risk** (>=70%): Red warning, recommend merging main first
- **MEDIUM risk** (40-70%): Yellow warning
- **LOW risk** (<40%): Green, safe to merge

## Configuration

```json
{
  "mergeConflictPredictor.serverUrl": "http://localhost:5000",
  "mergeConflictPredictor.autoCheck": true,
  "mergeConflictPredictor.riskThreshold": 0.5
}
```

## Architecture

```
VS Code Extension
├── src/extension.js        # Main entry point
├── src/gitMonitor.js       # Monitors branch changes
├── src/featureExtractor.js # Extracts git features
├── src/predictionClient.js # Calls prediction server
└── src/conflictPanel.js    # Webview panel

Prediction Server
├── prediction_server.py    # REST API server
├── conflict_predictor.py   # ML model
└── model.pkl              # Trained model
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Predict conflict |
| `/features` | GET | Get features for branches |

## Development

```bash
# Install dependencies
npm install

# Run linting
npm run lint

# Package extension
npm run package
```

## License

MIT
