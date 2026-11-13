# GitHub publication (perform on your Pop!_OS workstation)

The connected GitHub app can edit existing repositories but does not expose an action to create a new repository here; chat access does not expose your local terminal. You must run the following commands on your machine.

```bash
cd ~/projects/stellar-reproduction
# Unzip this update bundle into the current folder first.

# Bring only the analysis scripts you wrote locally (not the STELLAR upstream source).
bash scripts/copy_local_analysis_code.sh

# Add ignore rules before staging to avoid inadvertently committing raw results.
cat GITIGNORE_APPEND.txt >> .gitignore

# Verify existing runs and regenerate plots from original local CSV results.
python scripts/check_existing_results.py --results ~/projects/stellar/results
python make_figures.py --results ~/projects/stellar/results

# Review public-facing changes and any confidential/private content.
git status --short
find . -maxdepth 3 -type f -not -path './.git/*' | sort

# Do not commit data/checkpoints, human-subject raw material, or credentials.
git add README.md Research_summary.md PUBLISH_GITHUB.md .gitignore \
    make_figures.py scripts/ tables/ figures/
git commit -m 'Add reproducible STELLAR ablation audit and region-data inspection'
git branch -M main

# If the GitHub CLI is absent: sudo apt update && sudo apt install gh
gh auth status || gh auth login --hostname github.com --git-protocol https --web
# Creates an independent public repo, sets the origin, and pushes local commits.
gh repo create EDCAKDC/stellar-reproduction --public --source=. --remote=origin --push
```

If the repository already exists, instead:

```bash
git remote add origin https://github.com/EDCAKDC/stellar-reproduction.git
git push -u origin main
```

Repository URL after successful upload: https://github.com/EDCAKDC/stellar-reproduction

**No publication has occurred until `gh repo create` or `git push` finishes successfully.**

## Independent region validation: data inspection first

Download **B004_training_dryad.csv** from the publisher's Dryad page:
https://doi.org/10.5061/dryad.g4f4qrfrc

Then:

```bash
python scripts/inspect_b004_csv.py ~/Downloads/B004_training_dryad.csv
```

The audit reports column names and candidate region labels. It does **not** automatically construct a graph or train a model. Before training, choose and verify independent section/region identifiers, expert cell-type label column, x/y coordinates, marker feature columns, and colon-only grouping. No cross-region graph edges, no test-section labels during training, validation by distinct region, and one untouched held-out test region. Until such validation is performed, do not claim cross-section or independent-sample performance.
