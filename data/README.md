# Data

These files are **not committed** (large binaries / gitignored). Copy them from
the reference project or original sources before running the build steps.

Expected layout:

```
data/
├── dataset.csv                 # symptom -> disease training data
├── Symptom-severity.csv        # symptom -> severity weight
├── symptom_Description.csv     # disease descriptions
├── symptom_precaution.csv      # disease precautions
├── mendeley_symptoms.csv       # secondary dataset
└── encyclopedias/
    └── encyclopedia-of-medicine-*.pdf   # source for the PDF vector store
```

Reference copy lives at `~/Studio1119/healthcare-rag-reference/data/`.
