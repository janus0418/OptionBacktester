---
name: update-docs-and-commit
description: Update project documentation and create a git commit with all changes
argument-hint: "[commit message description]"
---

# Update Documentation and Commit

This command updates project documentation to reflect recent changes and creates a well-structured git commit. Use this for all commits to ensure docs stay in sync with code.

## Workflow

### Step 1: Analyze Changes

1. Run `git status` and `git diff --staged` to see what has changed
2. If nothing is staged, run `git diff` to see unstaged changes
3. Identify the nature of the changes:
   - New features or functionality
   - Bug fixes
   - Refactoring
   - Configuration changes

### Step 2: Update Documentation

Based on the changes identified, update relevant documentation files:

1. **docs/changelog.md** - Add an entry describing what changed
   - Include date, brief description, and any breaking changes

2. **docs/project_status.md** - Update if:
   - A milestone was completed
   - Progress was made on current features
   - New tasks were identified

3. **docs/architecture.md** - Update if:
   - New modules or components were added
   - Data flow changed
   - System design was modified

4. **docs/project_spec.md** - Update if:
   - Requirements changed
   - Technical specifications were modified

### Step 3: Stage All Changes

```bash
git add -A
```

### Step 4: Create Commit

Create a commit with the message based on #$ARGUMENTS:

- If no argument provided, analyze the changes and generate an appropriate commit message
- Follow conventional commit format when applicable (feat:, fix:, docs:, refactor:, etc.)
- Keep the first line under 72 characters
- Add body with details if the change is significant

### Step 5: Verify

1. Run `git log -1` to confirm the commit
2. Run `git status` to ensure working directory is clean

## Success Criteria

- [ ] All relevant documentation files are updated
- [ ] Changes are staged and committed
- [ ] Commit message clearly describes the change
- [ ] Working directory is clean after commit
