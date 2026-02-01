# Scripts (`scripts/`)

Operational and utility scripts organized by purpose.

## Structure

```
scripts/
├── deployment/      # HuggingFace Space & model deployment
├── colab/           # Google Colab notebooks/scripts
└── db_maintenance/  # Database utilities and cleanup
```

## Deployment

Scripts for deploying to HuggingFace:
- `app.py` - Gradio app for HF Spaces (test mode)
- `app_v2.py` - Enhanced app with mode selection
- `deploy_to_hf_space.ps1` - Deployment automation
- `README_SPACE.md` - Space documentation
- Various `push_*.ps1` scripts for fixes

## Colab

Google Colab code snippets:
- `COLAB_EVAL_CODE.py` - Model evaluation code
- `COLAB_PUSH_MODEL.py` - Push model to Hub
- `COLAB_TEST_MODEL.py` - Test model inference

## DB Maintenance

Database utilities for `runs.db`:
- `db_editor.py` - Interactive database editor
- `check_db_status.py` - Database health check
- `check_repo_status.py` - Repository status
- `delete_*.py` - Data cleanup scripts
- `find_missing_dataset.py` - Find missing datasets
