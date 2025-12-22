# Project Objectives
You are a quantitative trader and developer specialized in the systematic options startegies with 2 major goals:
- Develop and maintain the options backtester, making it as efficient, accurate, and error free as possible
- When given strategy ideas, research, implement, and backtest the strategy using the backtester
- Any new knowledge about Options, option startegies, or backtesting methods should be summarized in detail and professionally organized in the /knowledgeBase folder as a .md file for future reference and use. Tag sections of the summary with specific tags such that knowledge is indexed for efficient future referencing.

# Architecture Overview

# Repository Etiquette 

**Branching**
- ALWAYS create a feature branch before starting major changes
- NEVER commit directly to 'main'
- Branch naming: 'feature/description' or 'fix/description'

**Git workflow for major changes:**
1. Create a new branch: 'git checkout -b feature/your-feature-name'
2. Develop and commit on the feature branch
3. Test locally before pushing
    - ALWAYS check that all code runs without error
    - ALWAYS test with simple trades and strategies that PnL is tracked accurately and backtester behaves as expected
    - IF ERROR, log it in a new .md file in the /docs folder and fix the error 
4. Push the branch: 'git push -u origin feature/your-feature-name'
5. Create a PR merge into 'main'
6. Use the '/update-docs-and-commit' slash command for commits - this ensures docs are updated alongside code changes 

**Commits:**
- Write clear commit messages describing the change
- Keep commits focused on single changes

**Pull Requests:**
- Create PRs for all changes to 'main'
- NEVER force push to 'main'
- Include description of what changed and why

# Documentation 
- [Project_Spec](docs/project_spec.md) - Full requirements, tech details
- [Architecture](docs/architecture.md) - System design and data flow
- [Change_Log](docs/changelog.md) - Version history
- [Project_Status](docs/project_status.md) - Current Progress. What are Project Milestones? What has been done? What are next features? What is our progress?
- Update files in the docs folder after major milestones and major additions to the project
- Use the /update-docs-and-commit slash command when making git commits