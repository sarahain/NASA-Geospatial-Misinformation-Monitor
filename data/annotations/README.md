# Human labels (ground truth)

Put `manual_labels.csv` here with columns `post_id,human_label,notes`.

- `human_label` is one of `true_event`, `false_or_misleading`, `unclear`.
- Only `post_id` and the label are stored (no post text), so this file is
  tracked in Git.
- Generate a stratified sample to label with
  `python src/research_evaluation.py --make-label-template`
  (writes the git-ignored `data/processed/manual_labels_template.csv`, which
  does contain text for the labeller to read).
- Re-run `python src/research_evaluation.py` afterwards; precision / recall /
  F1 for NLP-only vs satellite-assisted appear in `data/evaluation/label_metrics.csv`
  and on the dashboard's Model Evaluation page.

Until this file exists, the project reports **no** accuracy figures.
