# explainable-recsys


---

## Development Workflow

We follow a **feature-branch + pull request workflow** with issue-based development:

1. **Issues**
   - Create a new GitHub Issue describing the task or feature.
   - Assign it to yourself or a team member.
   - Include a clear description, acceptance criteria, and priority.

2. **Branching**
   - Create a feature branch **from the issue**:
   - `main` – stable, production-ready code
   - `develop` – integration branch for feature branches
   - Feature branches: `feature/xxx` for any new feature or notebook

2. **Pull Requests**
   - All changes must go through a **pull request** to `develop` or `main`
   - PRs are automatically checked by **Ruff linter** via GitHub Actions

3. **Linting**
   - Ruff linter runs on every PR
   - Fix all reported issues before merging

4. **Merging**
   - Merge PRs only after code review and successful checks
   - Use `rebase and merge` to maintain linear history

### **Summary:** 

Start with an issue → create a branch from the issue → work → PR → review → merge. 

This ensures traceability, clean history, and quality control.

---

## Environment Setup

## 1. Creating a Virtual Environment

- https://code.visualstudio.com/docs/python/environments
- https://realpython.com/python-virtual-environments-a-primer/
- https://docs.python.org/3/library/venv.html

## 2. Activate the Environment

- Your terminal prompt should show (.venv) indicating the environment is active.
- Packages installed now will only affect this environment.

## 3. Install Dependencies

```pip install -r requirements.txt```
