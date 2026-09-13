"""
Prediction Server for VS Code Extension
Provides REST API for conflict prediction

Usage:  python prediction_server.py
Port:   5000
"""
import os
import sys
import warnings
import json
import subprocess
import numpy as np
import pandas as pd
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

warnings.filterwarnings("ignore")

# Load the trained model
import pickle
import conflict_predictor as cp

MODEL_PATH = "model.pkl"
GIT = cp.FEATURES


def load_model():
    """Load the trained model."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
            return data["model"]
    return None


model = load_model()


class PredictionHandler(BaseHTTPRequestHandler):
    """HTTP request handler for predictions."""

    def do_POST(self):
        """Handle POST requests."""
        if self.path == "/predict":
            self.handle_predict()
        else:
            self.send_error(404, "Not Found")

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/health":
            self.handle_health()
        elif self.path == "/features":
            self.handle_get_features()
        else:
            self.send_error(404, "Not Found")

    def handle_health(self):
        """Health check endpoint."""
        response = {"status": "ok", "model_loaded": model is not None}
        self.send_json(response)

    def handle_predict(self):
        """Handle prediction request."""
        try:
            # Read request body
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            features = json.loads(post_data.decode("utf-8"))

            # Extract features
            feature_vector = [
                features.get("commits_p1", 0),
                features.get("commits_p2", 0),
                features.get("files_p1", 0),
                features.get("files_p2", 0),
                features.get("overlap_files", 0),
                features.get("overlap_ratio", 0),
                features.get("authors_p1", 0),
                features.get("authors_p2", 0),
                features.get("churn_p1", 0),
                features.get("churn_p2", 0),
            ]

            X = np.array(feature_vector).reshape(1, -1)

            # Predict
            if model is not None:
                probability = float(model.predict_proba(X)[0][1])
            else:
                # Fallback: rule-based prediction
                probability = self.rule_based_predict(features)

            risk_level = self.get_risk_level(probability)

            response = {
                "probability": probability,
                "risk_level": risk_level,
                "features": features,
                "model_used": model is not None,
            }

            self.send_json(response)

        except Exception as e:
            self.send_error(500, str(e))

    def handle_get_features(self):
        """Get features for a merge scenario."""
        try:
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)

            repo = params.get("repo", [None])[0]
            branch_a = params.get("branchA", [None])[0]
            branch_b = params.get("branchB", [None])[0]

            if not repo or not branch_a or not branch_b:
                self.send_error(400, "Missing parameters")
                return

            # Extract features using git commands
            features = self.extract_features(repo, branch_a, branch_b)

            self.send_json(features)

        except Exception as e:
            self.send_error(500, str(e))

    def extract_features(self, repo, branch_a, branch_b):
        """Extract features from git repository."""

        def git_cmd(cmd):
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, cwd=repo
            )
            return result.stdout.strip()

        # Get merge base
        base = git_cmd(f"git merge-base {branch_a} {branch_b}")
        if not base:
            return None

        # Get commits
        commits_a = int(git_cmd(f"git rev-list --count {base}..{branch_a}") or 0)
        commits_b = int(git_cmd(f"git rev-list --count {base}..{branch_b}") or 0)

        # Get files
        files_a = set(git_cmd(f"git diff --name-only {base} {branch_a}").splitlines())
        files_b = set(git_cmd(f"git diff --name-only {base} {branch_b}").splitlines())
        files_a.discard("")
        files_b.discard("")

        # Calculate overlap
        overlap = files_a & files_b
        union = files_a | files_b
        overlap_ratio = len(overlap) / len(union) if union else 0

        # Get authors
        authors_a = len(
            set(git_cmd(f"git log --format=%ae {base}..{branch_a}").splitlines())
        )
        authors_b = len(
            set(git_cmd(f"git log --format=%ae {base}..{branch_b}").splitlines())
        )

        # Get churn
        def get_churn(branch):
            stat = git_cmd(f"git diff --shortstat {base} {branch}")
            ins = dels = 0
            for part in stat.split(", "):
                if "insertion" in part:
                    ins = int(part.split()[0])
                elif "deletion" in part:
                    dels = int(part.split()[0])
            return ins + dels

        churn_a = get_churn(branch_a)
        churn_b = get_churn(branch_b)

        return {
            "commits_p1": commits_a,
            "commits_p2": commits_b,
            "files_p1": len(files_a),
            "files_p2": len(files_b),
            "overlap_files": len(overlap),
            "overlap_ratio": overlap_ratio,
            "authors_p1": authors_a,
            "authors_p2": authors_b,
            "churn_p1": churn_a,
            "churn_p2": churn_b,
        }

    def rule_based_predict(self, features):
        """Rule-based prediction when model is not available."""
        score = 0

        # Overlap is strongest predictor
        if features.get("overlap_files", 0) > 0:
            score += features["overlap_files"] * 0.3

        # High churn increases risk
        if features.get("churn_p1", 0) > 100:
            score += 0.1
        if features.get("churn_p2", 0) > 100:
            score += 0.1

        # Many commits increase risk
        if features.get("commits_p1", 0) > 10:
            score += 0.05
        if features.get("commits_p2", 0) > 10:
            score += 0.05

        # Many authors increase risk
        if features.get("authors_p1", 0) > 3:
            score += 0.05
        if features.get("authors_p2", 0) > 3:
            score += 0.05

        # Overlap ratio
        score += features.get("overlap_ratio", 0) * 0.2

        return min(score, 1.0)

    def get_risk_level(self, probability):
        """Get risk level from probability."""
        if probability >= 0.7:
            return "HIGH"
        elif probability >= 0.4:
            return "MEDIUM"
        return "LOW"

    def send_json(self, data):
        """Send JSON response."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def send_error(self, code, message):
        """Send error response."""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode())

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def main():
    """Start the prediction server."""
    PORT = 5000

    print("=" * 60)
    print("  Merge Conflict Prediction Server")
    print("=" * 60)
    print(f"\n  Starting server on port {PORT}...")
    print(f"  Health check: http://localhost:{PORT}/health")
    print(f"  Predict:      http://localhost:{PORT}/predict")
    print(f"\n  Press Ctrl+C to stop\n")

    server = HTTPServer(("localhost", PORT), PredictionHandler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped")
        server.server_close()


if __name__ == "__main__":
    main()
