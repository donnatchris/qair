# Détection d'image avec yolo

## Commande principale

`yolo predict` lance YOLO en mode détection / prédiction sur une image ou une vidéo.

```bash
yolo predict <options>
```

### Options

Exemples des principales options:

- `model=yolo26m.pt` : utilise le modèle YOLO26 Medium, plus précis mais plus lourd que `n` ou `s`.
- `source=$(SOURCE)` : indique le fichier vidéo ou image à analyser.
- `imgsz=1280` : redimensionne les images d’entrée à 1280 px pour l’inférence, utile pour mieux voir les petits objets.
- `conf=0.05` : conserve les détections ayant au moins 5 % de confiance.
- `classes=14` : détécte uniquement les objets de la classe 14 (les oiseaux) - on sépatre par des virgules pour mettre plusieurs classes
- `device=cpu` : force l’exécution sur le processeur plutôt que sur un GPU.
- `save=True` : génère une copie de la vidéo avec les bounding boxes dessinées.
- `save_txt=True` : enregistre les détections de chaque frame dans des fichiers `.txt`.
- `save_conf=True` : ajoute le score de confiance de chaque détection dans les fichiers `.txt`.
