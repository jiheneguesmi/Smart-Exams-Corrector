# Exam OCR & Grading Pipeline

Système complet d'extraction de texte à partir d'examens scannés et notation automatique avec LLM open source via API.

## 📋 Fonctionnalités

- **OCR Handwritten** : Extraction de texte manuscrit à partir d'images (français/anglais)
- **Segmentation Q/A** : Séparation automatique questions/réponses
- **Notation LLM Open Source** : Évaluation avec modèles open source (Mistral, Llama-2, etc.) via API
- **Scores sur 20** : Notation sur l'échelle européenne (0-20)
- **Rapports détaillés** : Scores, feedback, recommandations d'amélioration
- **Traitement par lot** : Traite automatiquement tous les examens de `data/`

## 🗂️ Structure

```
projet asma/
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

## 📦 Installation

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

**Voir [LLM_SETUP.md](LLM_SETUP.md) pour le guide complet de configuration.**

## 🚀 Utilisation

### Traitement complet

```bash
python batch_process_exams.py
```

**Options** :
- `--data data` : Chemin vers le dossier data/ (défaut)
- `--ocr-model french` : Modèle OCR (`french`, `english`, `french_printed`)
- `--api-key your-key` : Clé API (ou utilise `LLM_API_KEY` env var)
- `--api-endpoint url` : URL de l'API (défaut: Hugging Face)
- `--model-name name` : Nom du modèle (défaut: Mistral-7B)
- `--skip-ocr` : Passe l'OCR si déjà traité

### Exemples

```bash
# Configuration minimale (utilise env var LLM_API_KEY)
export LLM_API_KEY="hf_your-hugging-face-key"
python batch_process_exams.py

# Avec clé en ligne de commande
python batch_process_exams.py --api-key "hf_your-key"

# Avec modèle Llama-2
python batch_process_exams.py --model-name "meta-llama/Llama-2-7b-chat-hf"

# Avec endpoint personnalisé (serveur local)
python batch_process_exams.py \
  --api-endpoint "http://localhost:8000/api/generate" \
  --model-name "votre-modele"

# Sans clé en ligne de commande (utilise env var)
export LLM_API_KEY="your-key"
python batch_process_exams.py
```

## 📊 Résultats

Les résultats sont sauvegardés dans `results/YYYYMMDD_HHMMSS/`:

```
results/20240115_143022/
├── summary.txt           # Résumé global
├── summary.json
├── GenAI/
│   └── copie1/
│       ├── 01_ocr_raw.txt       # Texte extrait
│       ├── 02_questions_answers.txt  # Q/A formatées
│       ├── 03_grade.json        # Note et feedback JSON
│       └── 04_report.txt        # Rapport lisible
│   └── copie2/
│       └── ...
└── MLOps/
    └── ...
```

### Contenu des fichiers

- **01_ocr_raw.txt** : Texte brut du OCR
- **02_questions_answers.txt** : Questions et réponses numérotées
- **03_grade.json** : 
  ```json
  {
    "score": 15.5,
    "grade_letter": "B",
    "strengths": ["Bonne compréhension conceptuelle", ...],
    "weaknesses": ["Manque de détails", ...],
    "feedback": "Analyse générale",
    "improvements": ["Approfondir X", ...]
  }
  ```
- **04_report.txt** : Rapport formaté pour l'étudiant

## 🤖 Modèles disponibles

### OCR
- `french` : Handwriting français (recommandé)
- `english` : Handwriting anglais
- `french_printed` : Texte imprimé français

### LLM Open Source

| Modèle | Vitesse | Qualité | Notes |
|--------|---------|---------|-------|
| **Mistral-7B** | ⚡⚡⚡ | ⭐⭐⭐⭐ | Excellent - recommandé |
| **Llama-2-7B** | ⚡⚡ | ⭐⭐⭐ | Bon |
| **Qwen-7B** | ⚡⚡⚡ | ⭐⭐⭐⭐ | Très bon |
| **OpenHermes-2.5** | ⚡⚡ | ⭐⭐⭐⭐ | Excellent |
| **Llama-2-13B** | ⚡ | ⭐⭐⭐⭐ | Haute qualité |
| **Llama-2-70B** | 🐢 | ⭐⭐⭐⭐⭐ | Meilleure qualité |

**Services recommandés:**
- **Hugging Face** (gratuit) : https://huggingface.co/
- **Together AI** (gratuit au départ) : https://www.together.ai/
- **Replicate** (gratuit + payant) : https://replicate.com/

## ⚙️ Configuration avancée

Pour modifier les paramètres d'OCR, éditez `batch_process_exams.py`:

```python
config = OCRConfig(
    model_type=ModelType.FRENCH,
    remove_watermark=True,           # CamScanner logos
    remove_blue_lines=True,          # Papier réglé
    max_line_height=70,              # Adapter si lignes fusionnent
    num_beams=6,                     # 8-10 pour plus de précision (plus lent)
)
```

## 🐛 Troubleshooting

### Clé API invalide
```
❌ OPENAI_API_KEY not found
```
→ Vérifiez votre clé API dans les variables d'env
→ Créez un fichier `.env` avec vos clés

### Erreur de connexion API
→ Vérifiez votre connexion Internet
→ Vérifiez que le provider est accessible

### OCR donne du charabia
→ Vérifiez que les images sont claires et bien scannées
→ Essayez `--ocr-model french_printed` si texte imprimé

### LLM lent
→ Utilisez un modèle plus léger (gpt-3.5 au lieu de gpt-4)
→ Ou `--skip-ocr` pour passer la notation

## 📝 Exemple de workflow

1. **Scannez vos examens** : CamScanner ou photographiez chaque page
2. **Organisez** : 
   ```
   data/GenAI/copie1/page1.jpg
   data/GenAI/copie1/page2.jpg
   data/GenAI/copie2/page1.jpg
   ...
   ```
3. **Lancez le script** :
   ```bash
   python batch_process_exams.py
   ```
4. **Consultez les résultats** dans le dossier `results/`

## 🔍 Format des noms de fichiers

Les fichiers images doivent être nommés simplement :
- Acceptés : `page1.jpg`, `1.png`, `answer.jpg`
- Rejetés : aucun fichier image = étudiant ignoré

## 📧 Fichiers de sortie JSON

Pour intégration avec d'autres systèmes :

```python
import json

# Lire le résumé
with open("results/20240115_143022/summary.json") as f:
    summary = json.load(f)
    for student_id, result in summary["by_subject"]["GenAI"]["students"].items():
        print(f"{student_id}: {result['grade']}/20")

# Lire la note détaillée
with open("results/20240115_143022/GenAI/copie1/03_grade.json") as f:
    grade = json.load(f)
    print(f"Score: {grade['score']}/20")
    print(f"Points forts: {grade['strengths']}")
```

## 🛠️ Personnalisation

### Changer le modèle OCR

Éditez `batch_process_exams.py`, ligne ~45:

```python
OCRConfig(
    model_type=ModelType.ENGLISH,  # ← ENGLISH, FRENCH, ou FRENCH_PRINTED
)
```

### Ajouter de nouveaux modèles

Modifiez `ocr_pipeline.py`, classe `TrOCREngine`:

```python
MODEL_CONFIGS = {
    ModelType.CUSTOM: {
        "processor": "microsoft/trocr-large-handwritten",
        "model": "votre_modele_huggingface",
    }
}
```

### Personnaliser la notation

Modifiez le prompt dans `grader.py`, méthode `_build_prompt()`.

## 📖 Références

- **TrOCR** (Handwriting recognition) : https://huggingface.co/microsoft/trocr-large-handwritten
- **Ollama** (Local LLM) : https://ollama.ai
- **OpenCV** (Image processing) : https://docs.opencv.org

## 📄 Licence

Libre d'utilisation pour fins éducatives.

## 🤝 Support

Pour des problèmes ou améliorations, contactez-moi ou consultez les logs détaillés.
