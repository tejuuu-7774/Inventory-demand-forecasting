# Phase 2 model artifacts

Only artifacts whose names end in `_v2` belong to the corrected evaluation.
Pre-remediation manifests and checkpoints were removed during final repository
cleanup; their invalidation is documented in the remediation report.

The JSON files record model configuration, features, category mappings, split dates, training examples, device and runtime.

Binary `.pt` checkpoints are generated locally and intentionally ignored by Git. Recreate them by executing notebooks 04–07 in order with the isolated Phase 2 environment.
