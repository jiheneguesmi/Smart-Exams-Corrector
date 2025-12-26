# Exam OCR & Grading Pipeline

Système complet d'extraction de texte à partir d'examens scannés et notation automatique avec LLM open source via API.

##  Fonctionnalités

- **OCR Handwritten** : Extraction de texte manuscrit à partir d'images (français/anglais)
- **Segmentation Q/A** : Séparation automatique questions/réponses
- **Notation LLM Open Source** : Évaluation avec modèles open source (Mistral, Llama-2, etc.) via API
- **Scores sur 20** : Notation sur l'échelle européenne (0-20)
- **Rapports détaillés** : Scores, feedback, recommandations d'amélioration
- **Traitement par lot** : Traite automatiquement tous les examens de `data/`

##  Structure

```

├── data/
│   ├── GenAI/          # Matière 1
│   │   ├── copie1/     # Examen étudiant 1
│   │   ├── copie2/     # Examen étudiant 2
│   │   └── ...
│   └── MLOps/          # Matière 2
│       ├── copie1/
│       ├── copie2/
│       └── ...
├── results/            # Résultats (créé automatiquement)
├── ocr_pipeline.py     # Module OCR
├── qa_extractor.py     # Segmentation Q/A
├── grader.py           # Notation avec LLM
├── batch_process_exams.py  # Script principal
└── requirements.txt
```

##  Installation

### 1. Dépendances Python

```bash
pip install -r requirements.txt
```

**GPU recommandé** : Si vous avez NVIDIA (CUDA), TorchVision utilisera GPU pour accélérer l'OCR (~10x).

### 2. Configuration API LLM Open Source

Le système utilise des modèles LLM open source via API (Hugging Face par défaut).

**Étape 1 : Obtenir une clé API**

Choisissez un service :
- **Hugging Face** (recommandé) : https://huggingface.co/settings/tokens
- **Together AI** : https://api.together.xyz/
- **Replicate** : https://replicate.com/
- **Ou utilisez votre propre serveur local**

**Étape 2 : Configurer la clé**

Option A - Variable d'environnement (Windows PowerShell):
```powershell
$env:LLM_API_KEY = "your-api-key-here"
```

Option B - Variable d'environnement (Mac/Linux):
```bash
export LLM_API_KEY="your-api-key-here"
```

Option C - Fichier `.env` (à la racine du projet):
```
LLM_API_KEY=your-api-key-here
```

Option D - En ligne de commande:
```bash
python batch_process_exams.py --api-key "your-api-key-here"
```


