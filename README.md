# Clo — Transcription audio locale

Mini-CLI Python pour transcrire des fichiers audio (réunions, entretiens, etc.) en local avec [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Optionnellement, sépare les locuteurs (`Speaker 1`, `Speaker 2`, ...) via [pyannote.audio](https://github.com/pyannote/pyannote-audio).

100% local, gratuit, hors-ligne (sauf 1er téléchargement des modèles).

## Installation

```bash
# 1. Créer l'environnement virtuel et installer les dépendances
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. (Optionnel) configurer le token HuggingFace pour la diarisation
cp .env.example .env
# puis éditer .env et coller ton token HF_TOKEN
```

`ffmpeg` doit être disponible sur le système (`brew install ffmpeg` sur macOS).

## Utilisation

```
.
├── input/      ← déposer les fichiers audio ici
└── output/     ← un sous-dossier par fichier (txt + srt)
```

### Transcription simple

```bash
python transcribe.py -l fr
```

Tous les fichiers audio de `input/` sont transcrits. Pour `reunion.mp3`, ça produit :
- `output/reunion/reunion.txt`
- `output/reunion/reunion.srt`

Les fichiers déjà transcrits sont sautés automatiquement.

### Transcription avec diarisation

```bash
python transcribe.py -l fr --diarize --vad
```

Le `.txt` est regroupé par locuteur :
```
Speaker 1:
  Bonjour à tous, on commence ?

Speaker 2:
  Oui, c'est bon.
```

Le `.srt` préfixe chaque sous-titre par `[Speaker X]`.

### Options

| Option | Défaut | Description |
| --- | --- | --- |
| `-i`, `--input-dir` | `input` | Dossier des fichiers audio à traiter |
| `-o`, `--output-dir` | `output` | Dossier des transcriptions |
| `-m`, `--model` | `small` | Modèle Whisper : `tiny`, `base`, `small`, `medium`, `large-v3` |
| `-l`, `--language` | auto | Code langue (`fr`, `en`, ...) |
| `--device` | `auto` | `auto`, `cpu`, `cuda` |
| `--compute-type` | `int8` | Type de calcul (`int8`, `float16`, `float32`) |
| `--vad` | off | Active le filtre VAD (saute les silences) |
| `--diarize` | off | Active la diarisation (nécessite un HF_TOKEN) |
| `--hf-token` | env | Token HuggingFace (sinon lu depuis `.env` / env) |

### Choix du modèle

| Modèle | Taille | Vitesse CPU | Qualité |
| --- | --- | --- | --- |
| `tiny` | 75 Mo | ~10x temps réel | Faible |
| `base` | 150 Mo | ~7x temps réel | Correcte |
| `small` | 500 Mo | ~4x temps réel | Bonne (défaut) |
| `medium` | 1.5 Go | ~2x temps réel | Très bonne |
| `large-v3` | 3 Go | ~1x temps réel | Excellente |

Sur Mac Apple Silicon, `medium` est un bon compromis pour des fichiers de 30min–1H.

## Activer la diarisation (configuration HuggingFace)

À faire **une seule fois** :

1. Créer un compte sur https://huggingface.co
2. Accepter les conditions des trois modèles :
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-community-1
3. Générer un token (type *Read*) sur https://huggingface.co/settings/tokens
4. Le coller dans `.env` (variable `HF_TOKEN`)

## Performances

Sur Mac M1/M2 (CPU, modèle `small`), comptez environ :
- Transcription seule : ~0.5x du temps audio (1h d'audio → 30min)
- Avec `--diarize` : ajoutez 0.5x à 1x supplémentaire

## Structure du projet

```
.
├── transcribe.py       # CLI
├── requirements.txt
├── .env                # secrets locaux (ignoré par git)
├── .env.example        # template
├── .gitignore
├── input/              # fichiers audio à traiter
└── output/             # transcriptions générées
```
