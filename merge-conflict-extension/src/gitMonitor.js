const { exec } = require('child_process');
const EventEmitter = require('events');

/**
 * Monitors Git repository for branch changes
 */
class GitMonitor extends EventEmitter {
    constructor() {
        super();
        this.watcher = null;
        this.lastBranch = null;
        this.checkInterval = null;
    }

    /**
     * Start monitoring git changes
     */
    start() {
        // Check branch every 5 seconds
        this.checkInterval = setInterval(() => {
            this.checkBranchChange();
        }, 5000);

        console.log('Git monitor started');
    }

    /**
     * Stop monitoring
     */
    stop() {
        if (this.checkInterval) {
            clearInterval(this.checkInterval);
            this.checkInterval = null;
        }
        console.log('Git monitor stopped');
    }

    /**
     * Check if branch changed
     */
    async checkBranchChange() {
        const workspaceRoot = this.getWorkspaceRoot();
        if (!workspaceRoot) return;

        const currentBranch = await this.getCurrentBranch(workspaceRoot);
        if (!currentBranch) return;

        if (this.lastBranch && this.lastBranch !== currentBranch) {
            this.emit('branchChange', {
                from: this.lastBranch,
                to: currentBranch,
                branch: currentBranch
            });
        }

        this.lastBranch = currentBranch;
    }

    /**
     * Get current branch name
     */
    async getCurrentBranch(repoPath) {
        return new Promise((resolve) => {
            exec('git branch --show-current', { cwd: repoPath }, (error, stdout) => {
                if (error) {
                    resolve(null);
                    return;
                }
                resolve(stdout.trim());
            });
        });
    }

    /**
     * Get workspace root path
     */
    getWorkspaceRoot() {
        const vscode = require('vscode');
        const folders = vscode.workspace.workspaceFolders;
        return folders?.[0]?.uri?.fsPath || null;
    }

    /**
     * Get list of branches
     */
    async getBranches(repoPath) {
        return new Promise((resolve) => {
            exec('git branch -a --format=%(refname:short)', { cwd: repoPath }, (error, stdout) => {
                if (error) {
                    resolve([]);
                    return;
                }
                const branches = stdout.trim().split('\n')
                    .filter(b => !b.includes('HEAD') && !b.includes('remotes/'));
                resolve(branches);
            });
        });
    }

    /**
     * Check if working directory is clean
     */
    async isWorkingDirectoryClean(repoPath) {
        return new Promise((resolve) => {
            exec('git status --porcelain', { cwd: repoPath }, (error, stdout) => {
                resolve(stdout.trim() === '');
            });
        });
    }
}

module.exports = { GitMonitor };
