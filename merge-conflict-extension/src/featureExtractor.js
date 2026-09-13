const { exec } = require('child_process');

/**
 * Extracts git-history features for merge conflict prediction
 */
class FeatureExtractor {
    /**
     * Extract features for a merge scenario
     */
    async extract(repoPath, branchA, branchB) {
        try {
            // Get merge base
            const base = await this.getMergeBase(repoPath, branchA, branchB);
            if (!base) {
                throw new Error('Could not find merge base');
            }

            // Get parent commits
            const commitA = await this.getBranchCommit(repoPath, branchA);
            const commitB = await this.getBranchCommit(repoPath, branchB);

            // Extract features for both branches
            const featuresA = await this.extractBranchFeatures(repoPath, base, commitA);
            const featuresB = await this.extractBranchFeatures(repoPath, base, commitB);

            // Calculate overlap
            const filesA = await this.getChangedFiles(repoPath, base, commitA);
            const filesB = await this.getChangedFiles(repoPath, base, commitB);
            const overlap = this.calculateOverlap(filesA, filesB);

            // Detect conflict
            const conflict = await this.detectConflict(repoPath, commitA, commitB);

            return {
                repo: this.getRepoName(repoPath),
                merge: 'pending',
                commits_p1: featuresA.commits,
                commits_p2: featuresB.commits,
                files_p1: featuresA.fileCount,
                files_p2: featuresB.fileCount,
                overlap_files: overlap.count,
                overlap_ratio: overlap.ratio,
                authors_p1: featuresA.authors,
                authors_p2: featuresB.authors,
                churn_p1: featuresA.churn,
                churn_p2: featuresB.churn,
                conflict_files: conflict.files.join(';'),
                code_conflict: conflict.hasCodeConflict ? 1 : 0,
                label: conflict.hasConflict ? 1 : 0
            };
        } catch (error) {
            console.error('Feature extraction failed:', error);
            throw error;
        }
    }

    /**
     * Get merge base of two branches
     */
    async getMergeBase(repoPath, branchA, branchB) {
        return new Promise((resolve) => {
            const cmd = `git merge-base ${branchA} ${branchB}`;
            exec(cmd, { cwd: repoPath }, (error, stdout) => {
                if (error) {
                    resolve(null);
                    return;
                }
                resolve(stdout.trim());
            });
        });
    }

    /**
     * Get commit hash for a branch
     */
    async getBranchCommit(repoPath, branch) {
        return new Promise((resolve) => {
            exec(`git rev-parse ${branch}`, { cwd: repoPath }, (error, stdout) => {
                if (error) {
                    resolve(null);
                    return;
                }
                resolve(stdout.trim());
            });
        });
    }

    /**
     * Extract features for one branch
     */
    async extractBranchFeatures(repoPath, base, tip) {
        const commits = await this.getCommitCount(repoPath, base, tip);
        const files = await this.getChangedFiles(repoPath, base, tip);
        const authors = await this.getAuthorCount(repoPath, base, tip);
        const churn = await this.getChurn(repoPath, base, tip);

        return {
            commits: commits,
            fileCount: files.length,
            files: files,
            authors: authors,
            churn: churn
        };
    }

    /**
     * Get commit count between two points
     */
    async getCommitCount(repoPath, base, tip) {
        return new Promise((resolve) => {
            exec(`git rev-list --count ${base}..${tip}`, { cwd: repoPath }, (error, stdout) => {
                if (error) {
                    resolve(0);
                    return;
                }
                resolve(parseInt(stdout.trim()) || 0);
            });
        });
    }

    /**
     * Get list of changed files
     */
    async getChangedFiles(repoPath, base, tip) {
        return new Promise((resolve) => {
            exec(`git diff --name-only ${base} ${tip}`, { cwd: repoPath }, (error, stdout) => {
                if (error) {
                    resolve([]);
                    return;
                }
                const files = stdout.trim().split('\n').filter(f => f);
                resolve(files);
            });
        });
    }

    /**
     * Get distinct author count
     */
    async getAuthorCount(repoPath, base, tip) {
        return new Promise((resolve) => {
            exec(`git log --format=%ae ${base}..${tip}`, { cwd: repoPath }, (error, stdout) => {
                if (error) {
                    resolve(0);
                    return;
                }
                const authors = new Set(stdout.trim().split('\n').filter(a => a));
                resolve(authors.size);
            });
        });
    }

    /**
     * Get churn (lines added + deleted)
     */
    async getChurn(repoPath, base, tip) {
        return new Promise((resolve) => {
            exec(`git diff --shortstat ${base} ${tip}`, { cwd: repoPath }, (error, stdout) => {
                if (error) {
                    resolve(0);
                    return;
                }
                let insertions = 0;
                let deletions = 0;

                const parts = stdout.split(', ');
                for (const part of parts) {
                    if (part.includes('insertion')) {
                        insertions = parseInt(part) || 0;
                    } else if (part.includes('deletion')) {
                        deletions = parseInt(part) || 0;
                    }
                }

                resolve(insertions + deletions);
            });
        });
    }

    /**
     * Calculate overlap between two file sets
     */
    calculateOverlap(filesA, filesB) {
        const setA = new Set(filesA);
        const setB = new Set(filesB);
        const intersection = [...setA].filter(f => setB.has(f));
        const union = new Set([...setA, ...setB]);

        return {
            count: intersection.length,
            ratio: union.size > 0 ? intersection.length / union.size : 0,
            files: intersection
        };
    }

    /**
     * Detect if merge will conflict
     */
    async detectConflict(repoPath, commitA, commitB) {
        return new Promise((resolve) => {
            exec(
                `git merge-tree --write-tree ${commitA} ${commitB}`,
                { cwd: repoPath },
                (error, stdout, stderr) => {
                    const hasConflict = error !== null || stdout.includes('CONFLICT');
                    const conflictFiles = [];

                    if (hasConflict) {
                        const lines = stdout.split('\n');
                        for (const line of lines) {
                            if (line.includes('Merge conflict in ')) {
                                const file = line.split('Merge conflict in ')[1];
                                if (file) {
                                    conflictFiles.push(file.trim());
                                }
                            }
                        }
                    }

                    const hasCodeConflict = conflictFiles.some(f =>
                        f.endsWith('.java') || f.endsWith('.py') ||
                        f.endsWith('.js') || f.endsWith('.ts')
                    );

                    resolve({
                        hasConflict: hasConflict,
                        hasCodeConflict: hasCodeConflict,
                        files: conflictFiles
                    });
                }
            );
        });
    }

    /**
     * Get repository name from path
     */
    getRepoName(repoPath) {
        return repoPath.split('/').pop().split('\\').pop();
    }
}

module.exports = { FeatureExtractor };
