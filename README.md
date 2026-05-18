# Clo — Transcription audio locale

Mini-app Python (CLI **et** interface web) pour transcrire des fichiers audio (réunions, entretiens, etc.) en local avec [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Optionnellement, sépare les locuteurs (`Speaker 1`, `Speaker 2`, ...) via [pyannote.audio](https://github.com/pyannote/pyannote-audio) puis permet de leur donner un vrai nom.

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

> Aucune dépendance externe à installer : `faster-whisper` embarque son propre décodeur audio (PyAV), donc pas besoin d'installer `ffmpeg` séparément.

## Interface web

```bash
python server.py
```

Puis ouvre http://127.0.0.1:8000

L'interface permet :
- de glisser un fichier audio
- de choisir langue, modèle, diarisation, VAD
- de voir la transcription apparaître **en direct** au fur et à mesure (segments + timestamps + locuteurs)
- (si diarisation) d'**écouter un échantillon de chaque locuteur** et de leur donner un vrai nom
- de **corriger manuellement** le locuteur d'une ligne mal attribuée (icône ✎ au hover sur chaque ligne)
- de **modifier directement le texte** d'un segment (clic sur le texte → édition inline, *Entrée* pour valider, *Échap* pour annuler)
- d'**écouter un passage précis** en cliquant sur son timestamp (saut + lecture automatique dans le mini-player audio en haut)
- de télécharger en **Word (.docx)**, **markdown**, `.txt`, `.srt` ou un `.zip` (txt + srt)

### Robustesse pour les longs fichiers

La transcription d'un fichier d'1h prend du temps. L'app est conçue pour survivre aux aléas :

- **PC qui se verrouille** : l'app demande un *Wake Lock* au navigateur pour empêcher la mise en veille pendant la transcription.
- **Onglet ou navigateur fermé puis rouvert** : l'`id` du job est mémorisé dans le `localStorage` ; à la prochaine ouverture, l'app se rebranche automatiquement sur le job qui continue côté serveur.
- **Connexion SSE qui tombe** : le worker tourne dans un thread de fond, **indépendamment** de la connexion. Une reconnexion rejoue tout l'historique des segments puis suit le live (déduplication automatique côté client).
- **Crash complet** : les fichiers `.txt`/`.srt` sont **écrits sur disque au fur et à mesure** dans `output/<nom>/` avec `flush()`. Même si le serveur Python crashe, ce qui a déjà été transcrit est sauvé.

## Utilisation en CLI

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
├── core.py              # logique partagée (whisper + pyannote + générateur de segments)
├── transcribe.py        # CLI batch (input/ -> output/)
├── server.py            # serveur FastAPI (interface web)
├── static/
│   ├── index.html       # interface web (HTML + CSS + JS, tout-en-un)
│   └── favicon.svg      # favicon fleur
├── requirements.txt
├── .env                 # secrets locaux (ignoré par git)
├── .env.example         # template
├── .gitignore
├── input/               # fichiers audio à traiter (mode CLI)
├── output/              # transcriptions générées (txt + srt, par sous-dossier)
└── uploads/             # uploads temporaires du mode web (ignoré par git)
```

## API du serveur (référence rapide)

| Méthode | Route | Description |
| --- | --- | --- |
| `GET` | `/` | Interface web |
| `POST` | `/upload` | Upload d'un fichier audio → retourne `job_id` |
| `POST` | `/start/{job_id}` | Lance le worker (en thread de fond) |
| `GET` | `/events/{job_id}` | SSE : rejoue l'historique puis suit le live |
| `GET` | `/job/{job_id}` | État d'un job (pour la reprise) |
| `GET` | `/sample/{job_id}/{speaker}.wav` | Échantillon vocal d'un locuteur |
| `POST` | `/rename/{job_id}` | Applique un mapping de noms (Speaker N → vrai nom) |
| `POST` | `/reassign/{job_id}` | Réassigne manuellement un segment à un autre locuteur |
| `POST` | `/edit/{job_id}` | Modifie le texte d'un segment |
| `GET` | `/audio/{job_id}` | Sert le fichier audio source (pour le mini-player + saut au timestamp) |
| `GET` | `/download/{job_id}/{txt\|srt\|md\|docx\|zip}` | Télécharge la transcription dans le format demandé |
