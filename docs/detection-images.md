# Comprendre la détection d'objets dans les images avec YOLO

Ce document présente les bases nécessaires pour comprendre comment une intelligence artificielle repère des objets dans une image, comment fonctionne une famille de modèles comme YOLO et comment interpréter ses résultats.

> **Idée centrale :** un détecteur d'objets ne « voit » pas une image comme un humain. Il reçoit une grille de nombres, y recherche des motifs appris pendant son entraînement, puis propose des rectangles, des classes et des scores de confiance.

---

## 1. La détection en une analogie

Imaginons un agent chargé de surveiller une grande photographie prise dans un aéroport. Pour chaque objet intéressant, il doit répondre à trois questions :

1. **Y a-t-il quelque chose ?**
2. **Qu'est-ce que c'est ?** Une personne, une valise, un véhicule…
3. **Où est-ce ?** En indiquant un rectangle autour de l'objet.

Un modèle de détection effectue le même travail. Pour chaque objet qu'il pense reconnaître, il produit généralement :

```text
classe       : oiseau
confiance    : 0,87
rectangle    : x1=412, y1=96, x2=468, y2=151
```

La confiance de `0,87` ne signifie pas que l'objet est « réellement un oiseau à 87 % » au sens mathématique strict. C'est un score produit par le modèle, utile pour classer et filtrer ses propositions. Sa fiabilité dépend de l'entraînement et peut nécessiter une calibration.

Analogie : un témoin peut dire « je suis presque certain d'avoir vu un oiseau ». Son assurance est une information utile, mais elle ne constitue pas une preuve.

---

## 2. Plusieurs problèmes de vision à ne pas confondre

L'expression « reconnaissance d'image » recouvre plusieurs tâches.

| Tâche | Question posée | Résultat typique |
|---|---|---|
| Classification | Que contient principalement l'image ? | `oiseau` |
| Localisation | Où se trouve l'objet principal ? | une classe et un rectangle |
| Détection d'objets | Quels objets sont présents et où ? | plusieurs classes et rectangles |
| Segmentation | Quels pixels appartiennent à chaque objet ? | un masque précis par objet |
| Estimation de pose | Où sont les points caractéristiques ? | articulations, yeux, ailes… |
| Suivi (*tracking*) | Quel objet est le même d'une image à l'autre ? | un identifiant persistant |

Une classification ressemble à une étiquette collée sur une boîte fermée. Une détection ressemble à un inventaire : « deux oiseaux ici, une personne là ». Une segmentation revient à découper précisément la silhouette de chaque objet avec des ciseaux.

YOLO est surtout connu pour la détection d'objets, mais certaines variantes peuvent également effectuer de la segmentation, de la classification, de l'estimation de pose ou d'autres tâches.

---

## 3. Ce que le modèle reçoit réellement

### 3.1 Une image est un tableau de nombres

Une image couleur est généralement représentée par trois valeurs par pixel : rouge, vert et bleu. Une image de 1920 × 1080 pixels peut donc être vue comme un tableau de forme :

```text
hauteur × largeur × canaux
1080    × 1920    × 3
```

Le modèle ne reçoit pas les concepts « ciel », « plume » ou « oiseau ». Il reçoit des nombres qui décrivent la couleur et l'intensité des pixels.

Avant l'analyse, le programme effectue habituellement plusieurs opérations :

- décoder le fichier ou la trame vidéo ;
- convertir ou réordonner les canaux de couleur ;
- redimensionner l'image vers une taille acceptée par le modèle ;
- compléter éventuellement les bords avec des bandes, opération appelée *letterboxing* ;
- normaliser les valeurs des pixels ;
- regrouper une ou plusieurs images dans un tenseur, appelé *batch*.

### 3.2 Redimensionner change l'information disponible

Supposons qu'un oiseau occupe 20 × 20 pixels dans une image 4K. Si l'image entière est réduite à 640 × 640, l'oiseau peut ne plus occuper que quelques pixels. Aucun modèle ne peut retrouver avec certitude des détails qui ont disparu pendant la réduction.

Analogie : observer une affiche depuis l'autre côté d'une rue. Même avec de très bons yeux, les petits caractères deviennent illisibles s'ils n'occupent plus assez de place sur la rétine.

Augmenter la taille d'entrée conserve davantage de détails, mais demande généralement :

- plus de calcul ;
- plus de mémoire ;
- plus de temps par image.

---

## 4. Comment un réseau de neurones apprend à regarder

Un détecteur moderne contient des millions de paramètres, appelés **poids**. Pendant l'entraînement, ces poids sont ajustés à partir d'images annotées.

```text
image d'entraînement
        +
annotations correctes
        ↓
prédiction du modèle
        ↓
mesure de l'erreur, appelée perte
        ↓
correction légère des poids
        ↓
répétition sur beaucoup d'exemples
```

Au début, les prédictions sont presque aléatoires. Après de nombreux exemples, certaines parties du réseau deviennent sensibles à des motifs utiles :

```text
premières couches     bords, contrastes, textures
        ↓
couches intermédiaires formes, motifs, parties d'objets
        ↓
couches profondes     combinaisons associées aux classes recherchées
```

Cette représentation est simplifiée : le réseau ne contient pas forcément un neurone réservé à « une aile » ou « un bec ». L'information est distribuée entre de nombreux calculs.

### Entraînement et inférence

Il faut distinguer deux phases :

- **l'entraînement** : le modèle apprend à partir d'images dont on connaît la bonne réponse ;
- **l'inférence** : le modèle déjà entraîné analyse de nouvelles images.

Analogie : l'entraînement correspond aux leçons et aux exercices corrigés. L'inférence correspond à l'examen, lorsque l'élève doit répondre sans disposer du corrigé.

Un fichier comme `yolo26m.pt` contient notamment des poids appris. Charger ce fichier ne réentraîne pas le modèle : cela restaure ce qu'il a déjà appris.

---

## 5. Les principales bibliothèques de vision et d'IA

YOLO est un modèle ou une famille de modèles. Pour l'utiliser dans une application, il faut aussi des bibliothèques capables de charger les images, d'exécuter les calculs et d'exploiter les résultats.

On peut comparer cet écosystème à un atelier :

```text
Python             langage utilisé pour donner les instructions
NumPy              établis sur lesquels sont rangés les nombres
PyTorch            machines qui construisent et entraînent le réseau
Ultralytics        poste de travail spécialisé qui simplifie YOLO
OpenCV             outils pour lire, découper, dessiner et écrire les images
CUDA / cuDNN       moteurs qui accélèrent certains calculs sur GPU NVIDIA
ONNX                format permettant de transporter le modèle vers un autre atelier
```

Ces outils se complètent. Ils ne font pas tous la même chose et il n'est pas nécessaire de tous les utiliser dans chaque projet.

### 5.1 Python : le chef d'orchestre

Python est le langage dans lequel le script du projet est écrit. Il organise les étapes :

```text
ouvrir la vidéo
    ↓
lire une trame
    ↓
demander une prédiction au modèle
    ↓
filtrer et enregistrer les résultats
```

Python n'effectue généralement pas lui-même les millions de multiplications nécessaires au réseau. Il appelle des bibliothèques écrites et optimisées dans des langages plus proches de la machine, puis récupère leurs résultats.

Analogie : le chef d'orchestre ne joue pas de tous les instruments. Il coordonne des spécialistes qui exécutent chacun leur partie.

### 5.2 NumPy : manipuler des tableaux de nombres

**NumPy** est une bibliothèque fondamentale du calcul scientifique en Python. Son objet principal, le tableau `ndarray`, permet de représenter efficacement des images et d'effectuer des opérations numériques.

Une image OpenCV en mémoire est généralement un tableau NumPy :

```python
height, width = frame.shape[:2]
tile = frame[y:y2, x:x2]
```

Ici, `frame` est un tableau et le découpage entre crochets en extrait une tuile sans avoir à manipuler les pixels un par un dans une boucle Python.

NumPy travaille principalement avec le CPU. Pour profiter facilement d'un GPU et calculer les gradients nécessaires à l'entraînement, on utilise plutôt un framework comme PyTorch.

### 5.3 PyTorch : construire, entraîner et exécuter le réseau

**PyTorch** est un framework de machine learning. Il fournit notamment :

- des **tenseurs**, c'est-à-dire des tableaux multidimensionnels proches des tableaux NumPy ;
- des opérations mathématiques optimisées pour le CPU et le GPU ;
- des composants permettant de construire les couches d'un réseau ;
- le calcul automatique des gradients utilisé pendant l'entraînement ;
- des outils pour charger les données, optimiser les poids et sauvegarder les modèles.

Un tenseur peut être déplacé vers un GPU compatible :

```python
image_cpu = torch.zeros((1, 3, 640, 640))
image_gpu = image_cpu.to("cuda")
```

Le même type d'opération mathématique peut ensuite être exécuté sur le CPU ou le GPU, à condition que l'installation et le matériel le permettent.

Pendant l'entraînement, PyTorch joue grossièrement le rôle suivant :

```text
prédiction
    ↓
calcul de la perte
    ↓
calcul automatique de l'influence de chaque poids
    ↓
mise à jour des poids par l'optimiseur
```

Analogie : le réseau est une immense table de réglages. PyTorch construit la machine, mesure dans quel sens tourner chaque bouton pour réduire l'erreur, puis permet d'appliquer ces corrections.

Dans le script du projet, PyTorch sert aussi à vérifier que CUDA et le GPU sont disponibles :

```python
torch.cuda.is_available()
torch.cuda.get_device_name(0)
```

Le fichier `.pt` est couramment associé à l'écosystème PyTorch. Il peut contenir les poids du modèle et d'autres informations nécessaires à son chargement. Ce suffixe ne décrit toutefois pas à lui seul tout le contenu exact du fichier.

### 5.4 Ultralytics : une interface de haut niveau pour YOLO

La bibliothèque **Ultralytics** fournit des modèles YOLO et une interface de haut niveau au-dessus du moteur de calcul. Elle évite d'avoir à reconstruire manuellement l'architecture, le prétraitement et le post-traitement.

Dans le projet :

```python
from ultralytics import YOLO

model = YOLO("yolo26m.pt")
results = model.predict(images, conf=0.05, device=0)
```

Ces quelques lignes déclenchent en réalité plusieurs étapes :

```text
charger l'architecture et les poids
    ↓
préparer les images
    ↓
exécuter le réseau avec PyTorch
    ↓
appliquer le post-traitement prévu
    ↓
retourner des objets contenant boîtes, classes et scores
```

La commande `yolo predict` est l'interface en ligne de commande du même écosystème. Elle est pratique pour essayer rapidement un modèle. L'API Python est plus adaptée lorsqu'il faut intégrer la détection dans une logique personnalisée, comme le découpage en tuiles.

Ultralytics et PyTorch ne sont donc pas deux concurrents directs dans ce projet : Ultralytics fournit la boîte à outils spécialisée YOLO, tandis que PyTorch effectue les opérations de réseau neuronal sous-jacentes.

### 5.5 OpenCV : travailler avec les images et les vidéos

**OpenCV** est une grande bibliothèque de vision par ordinateur. Elle ne se limite pas à l'IA et fournit notamment des fonctions pour :

- ouvrir une vidéo ou une caméra ;
- décoder et encoder des images ;
- redimensionner, recadrer ou convertir les couleurs ;
- dessiner des rectangles et du texte ;
- appliquer des algorithmes classiques de vision ;
- effectuer certaines formes de NMS.

Exemples tirés du type d'opérations utilisé dans le projet :

```python
cap = cv2.VideoCapture("video.mp4")  # ouvrir une vidéo
ok, frame = cap.read()               # lire une trame
cv2.rectangle(frame, p1, p2, color)  # dessiner une boîte
cv2.imwrite("resultat.jpg", frame)   # enregistrer une image
```

Analogie : PyTorch est le cerveau mathématique du détecteur ; OpenCV s'occupe plutôt de ses yeux, de ses feuilles de travail et de ses crayons.

Il faut aussi faire attention à l'ordre des couleurs : OpenCV représente couramment les images en **BGR**, alors que beaucoup d'autres outils parlent en **RGB**. Une bibliothèque de haut niveau effectue souvent la conversion attendue, mais du code personnalisé doit en tenir compte.

### 5.6 Torchvision, Pillow et Albumentations

D'autres bibliothèques sont courantes autour de PyTorch :

| Bibliothèque | Rôle principal |
|---|---|
| `torchvision` | jeux de données, transformations, modèles et opérations de vision compatibles avec PyTorch |
| `Pillow` (`PIL`) | ouverture, conversion et manipulation simple d'images |
| `Albumentations` | augmentations d'images et de boîtes pour enrichir les données d'entraînement |

Elles peuvent se recouvrir partiellement. Par exemple, OpenCV, Pillow et torchvision savent tous redimensionner une image, mais n'utilisent pas forcément le même type d'objet, le même ordre de couleurs ou les mêmes conventions de coordonnées.

Ajouter une bibliothèque n'améliore pas automatiquement la détection. Il faut choisir l'outil qui s'intègre le plus simplement au pipeline et limiter les conversions inutiles entre tableaux NumPy, images Pillow et tenseurs PyTorch.

### 5.7 ONNX et ONNX Runtime : transporter et exécuter le modèle

**ONNX** est avant tout un format d'échange décrivant un graphe de calcul et ses poids. Il permet d'exporter un modèle entraîné depuis un framework, puis de l'exécuter dans un autre environnement.

**ONNX Runtime** est un moteur capable d'exécuter de nombreux modèles au format ONNX sur différents matériels.

```text
modèle entraîné avec PyTorch
        ↓ export
fichier model.onnx
        ↓
ONNX Runtime, OpenCV DNN, TensorRT ou autre moteur compatible
```

Analogie : `.pt` ressemble à un appareil rangé dans la boîte de son fabricant. ONNX ressemble davantage à un plan standardisé permettant à plusieurs ateliers compatibles de reconstruire et d'exécuter le même mécanisme.

Un export n'est pas toujours parfaitement équivalent au modèle d'origine. Certaines opérations peuvent être converties différemment, et le prétraitement ou la NMS peuvent rester à la charge de l'application. Il faut donc comparer les sorties et les performances après export.

### 5.8 TensorFlow, Keras et JAX : d'autres frameworks

PyTorch n'est pas le seul framework de machine learning :

- **TensorFlow** fournit lui aussi tenseurs, entraînement, exécution et outils de déploiement ;
- **Keras** offre une interface de haut niveau pour construire et entraîner des modèles ;
- **JAX** propose un calcul numérique transformable, avec différentiation automatique et compilation, souvent utilisé en recherche.

Ces frameworks poursuivent des objectifs en partie similaires, avec des API et des écosystèmes différents. Un projet YOLO fondé sur Ultralytics et PyTorch n'a normalement pas besoin d'ajouter TensorFlow ou JAX pour effectuer la même inférence.

Analogie : ce sont plusieurs marques d'outillage capables de construire des machines comparables. Mélanger toutes les marques dans un même atelier n'est utile que si une pièce précise l'exige.

### 5.9 CUDA et cuDNN : sous les bibliothèques Python

**CUDA** est la plateforme de calcul parallèle de NVIDIA. **cuDNN** fournit des opérations optimisées pour les réseaux neuronaux. Elles se trouvent plus bas dans la pile logicielle que PyTorch ou Ultralytics :

```text
script Python
    ↓
Ultralytics
    ↓
PyTorch
    ↓
CUDA et bibliothèques accélérées
    ↓
pilote NVIDIA
    ↓
GPU
```

Le programme appelle généralement PyTorch, qui se charge lui-même d'appeler les couches CUDA appropriées. Utiliser `device=0` demande l'emploi du premier GPU visible ; cela ne suffit pas si le pilote, PyTorch ou ses dépendances GPU ne sont pas correctement installés.

Sur d'autres matériels, la pile d'accélération porte d'autres noms. Le rôle général reste le même : relier les opérations du framework aux capacités du processeur graphique ou de l'accélérateur.

### 5.10 Les utilitaires du script

Enfin, certaines bibliothèques ne font pas d'IA mais restent importantes pour construire une application complète :

| Outil | Utilisation dans le script |
|---|---|
| `pathlib` | construire et manipuler les chemins de fichiers |
| `json` | enregistrer les détections dans un format structuré |
| `datetime` | dater les dossiers de résultats |
| `tqdm` | afficher une barre de progression |

Un système de détection ne se résume donc pas au modèle. Il faut aussi lire les données, organiser le traitement, présenter la progression et sauvegarder des résultats exploitables.

---

## 6. Le principe de YOLO

YOLO signifie historiquement **You Only Look Once**. L'idée fondatrice est d'analyser l'image en une seule passe principale du réseau pour produire directement un ensemble de détections candidates.

Avant les détecteurs en une étape, certaines méthodes suivaient grossièrement ce parcours :

```text
1. chercher des régions qui pourraient contenir un objet
2. analyser séparément chaque région
3. classer et ajuster chaque rectangle
```

YOLO regroupe une grande partie de ce travail dans un même réseau :

```text
image
  ↓
extraction des caractéristiques
  ↓
combinaison de détails à plusieurs échelles
  ↓
prédiction de rectangles, classes et scores
  ↓
filtrage des propositions
```

Analogie : au lieu d'inspecter une salle avec une longue-vue, coin après coin, YOLO prend une vue d'ensemble et signale rapidement tous les éléments intéressants.

Cette approche est particulièrement adaptée aux images en temps réel et aux vidéos. « En une seule passe » ne signifie toutefois pas « en une seule opération » : le réseau exécute de nombreuses couches et beaucoup de calculs.

### Une famille, pas un unique modèle

Le nom YOLO désigne aujourd'hui une famille de modèles et d'implémentations qui ont évolué. Leur architecture exacte varie. Il faut donc éviter de transformer une explication historique — par exemple une grille rigide qui ne prédirait qu'un nombre fixe d'objets par cellule — en règle universelle pour toutes les versions modernes.

---

## 7. Les grandes parties d'un détecteur YOLO

Les noms précis diffèrent selon les versions, mais on peut retenir trois rôles conceptuels.

```text
Image
  │
  ▼
Backbone : extrait les caractéristiques visuelles
  │
  ▼
Neck : mélange les informations de plusieurs échelles
  │
  ▼
Head : prédit rectangles, classes et scores
  │
  ▼
Post-traitement : filtre et retire les doublons
```

### 7.1 Le *backbone* : prendre des notes

Le **backbone** transforme progressivement les pixels en cartes de caractéristiques. Il apprend des représentations utiles comme les contours, textures et formes.

Analogie : un enquêteur observe une scène et prend des notes structurées plutôt que de mémoriser chaque pixel.

### 7.2 Le *neck* : comparer plusieurs niveaux de zoom

Le **neck** combine souvent des informations à différentes résolutions :

- les cartes détaillées aident à repérer les petits objets ;
- les cartes plus compactes apportent davantage de contexte et aident pour les grands objets.

Analogie : sur une carte, une vue rapprochée montre les petites rues, tandis qu'une vue éloignée montre l'organisation générale de la ville. Les deux sont utiles.

### 7.3 La *head* : formuler les réponses

La **tête de détection** (*detection head*) produit les prédictions : position du rectangle, classe probable et score associé. Le réseau peut générer beaucoup plus de propositions que le nombre réel d'objets. Le post-traitement doit donc faire le tri.

---

## 8. Comprendre les rectangles englobants

Un **rectangle englobant**, ou *bounding box*, encadre approximativement un objet.

```text
(x1, y1) ┌─────────────────┐
         │                 │
         │     oiseau      │
         │                 │
         └─────────────────┘ (x2, y2)
```

Deux formats sont courants :

```text
coins       : x1, y1, x2, y2
centre      : x_centre, y_centre, largeur, hauteur
```

Les coordonnées peuvent être exprimées :

- en pixels, par exemple `x1 = 412` ;
- de façon normalisée entre 0 et 1, par exemple `x_centre = 0,54`.

Le format YOLO d'une annotation textuelle utilise souvent des coordonnées normalisées :

```text
class_id x_centre y_centre largeur hauteur
14 0.540 0.310 0.042 0.061
```

Normaliser revient à exprimer une position en proportion de l'image. Une coordonnée horizontale de `0,5` désigne le milieu, quelle que soit la largeur en pixels.

### Le rectangle n'est pas la silhouette

Une boîte contient souvent du fond autour de l'objet. Pour connaître précisément les pixels de l'objet, il faut un modèle de segmentation.

Analogie : ranger un objet dans un carton indique son emplacement et sa taille approximative, mais ne décrit pas sa forme exacte.

---

## 9. Le score de confiance et le seuil `conf`

Le modèle produit des candidats accompagnés d'un score. Le paramètre `conf` fixe le score minimum conservé.

```text
conf = 0,50

candidat A : oiseau, 0,91  → conservé
candidat B : oiseau, 0,63  → conservé
candidat C : oiseau, 0,18  → rejeté
```

Selon l'architecture et l'implémentation, ce score peut être calculé de différentes façons à partir de la présence probable d'un objet et de sa classe. Il vaut mieux le considérer comme un score de décision propre au modèle que comme une probabilité parfaitement calibrée.

### Baisser ou monter le seuil

| Seuil bas | Seuil haut |
|---|---|
| retrouve davantage d'objets difficiles | garde seulement les prédictions les plus sûres |
| produit généralement plus de faux positifs | risque de manquer davantage d'objets |
| utile pour privilégier le rappel | utile pour privilégier la précision |

Analogie : un filet de pêche aux mailles fines attrape davantage de poissons, mais aussi davantage de débris. Un filet plus sélectif produit une prise plus propre, au risque de laisser passer de petits poissons.

Il n'existe pas de valeur universelle idéale. Le bon seuil dépend du coût des deux erreurs :

- manquer un objet important ;
- déclencher une fausse alerte.

---

## 10. IoU : mesurer le chevauchement de deux boîtes

L'**Intersection over Union**, ou **IoU**, mesure le chevauchement entre deux rectangles.

```text
IoU = aire de l'intersection / aire de l'union
```

```text
IoU = 0     aucun chevauchement
IoU ≈ 0,5   chevauchement partiel
IoU = 1     rectangles identiques
```

Analogie : deux transparents colorés sont posés l'un sur l'autre. L'IoU compare la zone commune à toute la surface couverte par au moins l'un des transparents.

L'IoU sert principalement à deux moments :

1. pendant l'évaluation, pour décider si une prédiction localise correctement un objet réel ;
2. pendant le post-traitement, pour repérer des prédictions qui semblent désigner le même objet.

Il faut distinguer le seuil de confiance et le seuil d'IoU :

- `conf` demande « cette proposition est-elle assez crédible ? » ;
- `iou` demande « ces deux rectangles se recouvrent-ils assez pour être considérés comme des doublons ? ».

---

## 11. NMS : supprimer les doublons

Un détecteur peut proposer plusieurs rectangles autour du même objet. La **Non-Maximum Suppression**, ou **NMS**, conserve généralement le meilleur candidat et supprime les concurrents qui lui ressemblent trop.

```text
1. trier les boîtes par score
2. conserver la meilleure
3. supprimer les boîtes très chevauchantes de la même classe
4. recommencer avec les boîtes restantes
```

Analogie : plusieurs journalistes décrivent le même événement. On garde le témoignage le plus fiable et on évite de compter les autres récits comme des événements différents.

### Effet du seuil NMS IoU

- un seuil bas supprime plus facilement des boîtes voisines ;
- un seuil haut tolère davantage de chevauchement avant de supprimer une boîte.

Un seuil trop bas peut fusionner à tort deux objets proches. Un seuil trop haut peut laisser plusieurs boîtes sur le même objet.

Dans une scène contenant plusieurs oiseaux serrés les uns contre les autres, le choix est donc délicat : leurs rectangles peuvent naturellement se chevaucher sans être des doublons.

Certaines architectures ou bibliothèques modernes proposent aussi des variantes de NMS ou des sorties sans NMS classique. Le principe à retenir reste le même : transformer de nombreuses propositions en une liste finale cohérente.

---

## 12. Préparer les données d'entraînement

La qualité des données est souvent plus importante que le simple choix d'un modèle plus gros.

### 12.1 Annoter les images

Pour entraîner un détecteur, chaque objet utile doit généralement être associé à :

- une classe correcte ;
- un rectangle suffisamment précis ;
- une règle d'annotation cohérente avec le reste du jeu de données.

Par exemple, si les oiseaux partiellement cachés sont annotés dans certaines images mais ignorés dans d'autres, le modèle reçoit des consignes contradictoires.

Analogie : un professeur qui change les règles de correction à chaque copie empêche ses élèves de comprendre ce qui est attendu.

### 12.2 Séparer entraînement, validation et test

| Ensemble | Rôle |
|---|---|
| Entraînement | ajuster les poids du modèle |
| Validation | choisir les réglages et surveiller l'apprentissage |
| Test | mesurer une dernière fois la qualité sur des données jamais utilisées pour décider |

Ces ensembles doivent être réellement indépendants. Pour une vidéo, répartir au hasard des images voisines entre entraînement et validation est dangereux : deux trames consécutives sont presque identiques.

Une meilleure séparation peut consister à utiliser des vidéos, des jours, des caméras ou des lieux différents. Sinon, le modèle paraît excellent parce qu'il reconnaît le décor plutôt que l'objet.

### 12.3 Montrer la diversité du monde réel

Le jeu de données doit représenter les conditions d'utilisation :

- objets proches et lointains ;
- tailles, orientations et poses variées ;
- lumière, météo et arrière-plans différents ;
- objets partiellement cachés ;
- images sans objet cible ;
- cas difficiles susceptibles de créer de fausses alertes.

Les images négatives, qui ne contiennent aucun objet cible, apprennent au modèle ce qu'il ne faut pas détecter.

### 12.4 Augmentation de données

L'**augmentation** crée des variantes plausibles des images : retournement, recadrage, changement de luminosité, bruit, flou ou assemblage d'images selon les techniques utilisées.

Elle agit comme des exercices supplémentaires, mais elle doit rester réaliste. Retourner verticalement une scène peut être pertinent pour des objets sans orientation naturelle et absurde pour une caméra toujours fixée à l'endroit.

---

## 13. Ce qui se passe pendant l'entraînement

Quelques termes reviennent souvent :

| Terme | Signification |
|---|---|
| *batch* | petit groupe d'images traité avant une mise à jour des poids |
| époque | passage approximativement complet sur les données d'entraînement |
| perte (*loss*) | mesure différentiable des erreurs à réduire |
| taux d'apprentissage | taille des corrections appliquées aux poids |
| poids préentraînés | paramètres ayant déjà appris sur un grand jeu de données |
| *fine-tuning* | adaptation de poids préentraînés à un nouveau besoin |

La perte combine généralement plusieurs objectifs, par exemple :

- mieux placer et dimensionner les rectangles ;
- reconnaître la bonne classe ;
- distinguer objets et arrière-plan, selon l'architecture.

### Sous-apprentissage et surapprentissage

Le **sous-apprentissage** se produit lorsque le modèle n'a pas encore appris suffisamment, ou n'est pas capable de représenter le problème.

Le **surapprentissage** se produit lorsqu'il mémorise trop bien les particularités des données d'entraînement et généralise mal à de nouvelles images.

Analogie : un élève qui comprend la méthode peut résoudre un nouvel exercice. Un élève qui mémorise uniquement les réponses du manuel échoue dès que les nombres changent.

Signes possibles de surapprentissage : la qualité continue de monter sur l'entraînement, mais stagne ou baisse sur la validation.

---

## 14. Mesurer correctement la qualité

Observer quelques images annotées est utile, mais insuffisant. Il faut compter les résultats sur un jeu de données représentatif.

### 14.1 Vrais et faux résultats

| Résultat | Signification |
|---|---|
| Vrai positif (`TP`) | un objet réel est correctement détecté |
| Faux positif (`FP`) | le modèle annonce un objet qui ne devrait pas être compté |
| Faux négatif (`FN`) | un objet réel est manqué |

Pour décider qu'une détection correspond à la vérité terrain, on exige généralement la bonne classe et une IoU supérieure à un seuil choisi.

### 14.2 Précision et rappel

```text
précision = TP / (TP + FP)
rappel    = TP / (TP + FN)
```

- la **précision** répond : « parmi les alertes, combien étaient correctes ? » ;
- le **rappel** répond : « parmi les objets réels, combien ont été trouvés ? ».

Exemple :

```text
10 oiseaux réels
8 oiseaux correctement détectés
2 oiseaux manqués
4 fausses alertes

précision = 8 / (8 + 4) = 0,67
rappel    = 8 / (8 + 2) = 0,80
```

Le score **F1** combine précision et rappel lorsque l'on souhaite un compromis entre les deux.

### 14.3 AP et mAP

L'**Average Precision** (`AP`) résume la courbe précision-rappel d'une classe pour différentes valeurs du seuil de confiance. La **mean Average Precision** (`mAP`) calcule une moyenne sur plusieurs classes et, selon la métrique annoncée, sur un ou plusieurs seuils d'IoU.

Deux métriques fréquentes ne répondent pas exactement à la même question :

- `mAP@0.50` accepte une détection si l'IoU atteint au moins 0,50 ;
- `mAP@0.50:0.95` moyenne les résultats pour plusieurs seuils de 0,50 à 0,95 et exige donc aussi des localisations très précises.

Une mAP ne doit jamais être citée sans préciser sa définition, les classes évaluées et le jeu de données utilisé.

### 14.4 La métrique doit représenter le besoin réel

Pour un système d'alerte, on peut mesurer aussi :

- le nombre de fausses alertes par heure de vidéo ;
- la proportion d'événements réellement repérés ;
- le délai entre l'apparition d'un objet et sa détection ;
- la qualité selon la taille des objets ou les conditions météo.

Une excellente mAP globale peut cacher de mauvais résultats sur les petits oiseaux, précisément ceux qui intéressent l'application.

---

## 15. Pourquoi les petits objets sont difficiles

Un petit objet fournit peu de pixels. Il peut aussi disparaître partiellement lors des réductions successives effectuées par le réseau.

Plusieurs stratégies existent :

- utiliser une plus grande taille d'entrée ;
- entraîner avec davantage de petits objets correctement annotés ;
- choisir une architecture adaptée aux petites échelles ;
- améliorer la qualité optique et la résolution de la caméra ;
- découper l'image en tuiles ;
- ajuster le seuil de confiance avec une évaluation sérieuse.

Augmenter artificiellement une petite image ne recrée pas les détails absents. Cela peut faciliter certains calculs, mais ce n'est pas équivalent à capturer l'objet avec davantage de pixels réels.

### La détection par tuiles

Le script `scripts/detect_tiled.py` découpe chaque trame en zones plus petites :

```text
image complète
┌────────┬────────┬────────┐
│ tuile 1│ tuile 2│ tuile 3│
├────────┼────────┼────────┤
│ tuile 4│ tuile 5│ tuile 6│
└────────┴────────┴────────┘
```

Chaque tuile est agrandie ou adaptée à la taille d'entrée du modèle. Un petit oiseau occupe alors une part plus importante de l'image analysée.

Analogie : au lieu de regarder tout un paysage sur une petite carte postale, on l'examine morceau par morceau avec une loupe.

Cette méthode a plusieurs coûts :

- le modèle est exécuté plusieurs fois par trame ;
- le contexte global est réduit ;
- un objet coupé au bord d'une tuile peut être manqué ;
- le même objet peut être détecté dans plusieurs tuiles.

Le chevauchement entre tuiles réduit le risque de couper les objets. En contrepartie, il augmente encore le nombre de calculs et crée des doublons.

Le pipeline du projet peut se résumer ainsi :

```text
trame vidéo
  ↓
découpage en tuiles qui se chevauchent
  ↓
inférence YOLO sur chaque tuile
  ↓
conversion des coordonnées locales vers l'image complète
  ↓
NMS globale pour supprimer les doublons entre tuiles
  ↓
enregistrement des résultats
```

La NMS globale est importante : la NMS effectuée indépendamment dans chaque tuile ne peut pas savoir qu'une boîte de la tuile 1 et une boîte de la tuile 2 représentent le même oiseau.

---

## 16. Détecter dans une vidéo

Une vidéo est une suite d'images. Un détecteur classique analyse chaque trame indépendamment :

```text
frame 100 → 2 oiseaux
frame 101 → 2 oiseaux
frame 102 → 1 oiseau
```

Sans suivi, il ne sait pas automatiquement que l'oiseau de la frame 100 est le même que celui de la frame 101.

### Détection et suivi

Un algorithme de *tracking* associe les détections entre les trames et leur attribue des identifiants :

```text
frame 100 → oiseau #7
frame 101 → oiseau #7
frame 102 → oiseau #7
```

Cette distinction est essentielle pour compter des objets. Additionner toutes les détections de toutes les trames compte plusieurs fois le même oiseau.

Analogie : une caméra filme une personne pendant dix secondes. Le détecteur produit dix observations ; le tracker cherche à comprendre qu'elles appartiennent à une seule personne.

### Temps, fréquence et indices de trames

Pour une vidéo à `25 fps`, la frame 500 correspond approximativement à :

```text
temps = index / fps = 500 / 25 = 20 secondes
```

Ce calcul suppose une fréquence stable et des métadonnées correctes. Pour des vidéos à fréquence variable ou des besoins précis, les horodatages fournis par le conteneur vidéo sont plus fiables.

---

## 17. Vitesse, taille et précision

Il existe généralement plusieurs tailles d'un même modèle. Les suffixes varient selon la famille, mais suivent souvent une progression du type :

```text
petit modèle                         grand modèle
rapide ─────────────────────────────────► plus lent
léger  ─────────────────────────────────► plus lourd
moins de mémoire ───────────────────────► plus de mémoire
précision potentielle plus faible ──────► précision potentielle plus élevée
```

« Plus grand » ne garantit pas un meilleur résultat dans toutes les situations. La qualité des données, le domaine d'utilisation, la résolution et les seuils peuvent avoir davantage d'effet.

### Mesurer toute la chaîne

Le temps réel ne dépend pas seulement du réseau :

```text
lecture vidéo
  + décodage
  + prétraitement
  + transfert vers le GPU
  + inférence
  + NMS
  + dessin et écriture des résultats
```

Un modèle annoncé à 100 images par seconde peut produire une application beaucoup plus lente si le décodage, le découpage en tuiles ou l'enregistrement des images domine le temps total.

### Le rôle du GPU et du batch

Le GPU exécute efficacement de nombreux calculs similaires en parallèle. Un **batch** regroupe plusieurs images ou tuiles dans une même passe, ce qui peut améliorer le débit.

Cependant :

- un batch plus grand consomme plus de VRAM ;
- il peut augmenter le délai subi par une image individuelle ;
- le meilleur réglage dépend du matériel et de la taille des entrées.

Le document [`gpu.md`](gpu.md) explique plus en détail le rôle du GPU, de la VRAM et des transferts.

---

## 18. Lire les principaux paramètres du projet

La commande suivante illustre une inférence YOLO :

```bash
yolo predict \
    model=yolo26m.pt \
    source=videos/test-qair.mp4 \
    imgsz=1280 \
    conf=0.05 \
    classes=14 \
    device=cpu \
    save=True
```

| Paramètre | Question à laquelle il répond |
|---|---|
| `model` | Quels poids et quelle architecture utiliser ? |
| `source` | Quelle image, vidéo ou source analyser ? |
| `imgsz` | À quelle résolution le modèle doit-il travailler ? |
| `conf` | Quel score minimum conserver ? |
| `classes` | Quelles classes garder dans les résultats ? |
| `device` | Exécuter sur quel CPU ou GPU ? |
| `save` | Enregistrer une visualisation des résultats ? |

Dans `scripts/detect_tiled.py`, d'autres paramètres contrôlent la stratégie par tuiles :

| Paramètre | Rôle | Compromis principal |
|---|---|---|
| `TILE_SIZE` | taille d'une tuile en pixels | détail local contre contexte et nombre de tuiles |
| `OVERLAP` | recouvrement entre tuiles | objets moins coupés contre calcul supplémentaire |
| `BATCH_SIZE` | nombre de tuiles traitées ensemble | débit contre consommation de VRAM |
| `CONF` | score minimal | rappel contre fausses alertes |
| `NMS_IOU` | chevauchement toléré avant suppression | doublons contre objets proches fusionnés |
| `BIRD_CLASS` | identifiant de la classe recherchée | doit correspondre aux classes du modèle chargé |

L'identifiant numérique d'une classe n'est pas universel. `14` peut désigner `bird` dans un modèle entraîné avec une certaine liste de classes et autre chose dans un modèle personnalisé. Il faut toujours vérifier la table `model.names` associée aux poids.

---

## 19. Une méthode simple pour améliorer un détecteur

Lorsqu'un résultat semble mauvais, changer plusieurs réglages au hasard rend le diagnostic difficile. Une démarche plus fiable consiste à :

1. **Définir le besoin.** Faut-il éviter les oublis ou les fausses alertes ? Quelle taille minimale d'objet compte ?
2. **Créer un jeu d'évaluation représentatif.** Inclure les cas faciles, difficiles et négatifs.
3. **Mesurer un point de départ.** Précision, rappel, faux positifs par heure et temps de traitement.
4. **Regarder les erreurs.** Séparer petits objets, objets coupés, confusion de classes, mauvais éclairage et doublons.
5. **Changer une famille de paramètres à la fois.** Seuil, résolution, tuilage, modèle ou données.
6. **Mesurer de nouveau sur le même jeu.** Une impression visuelle ne suffit pas.
7. **Tester sur des données réellement nouvelles.** Vérifier que l'amélioration généralise.

Exemples de diagnostic :

| Symptôme | Causes possibles | Pistes à tester |
|---|---|---|
| petits oiseaux manqués | trop peu de pixels, données insuffisantes | résolution, tuiles, nouvelles annotations |
| beaucoup de faux oiseaux | seuil bas, arrière-plans absents de l'entraînement | seuil, images négatives, *fine-tuning* |
| plusieurs boîtes par oiseau | NMS trop permissive, doublons entre tuiles | NMS globale, seuil IoU |
| deux oiseaux deviennent une boîte | NMS trop agressive, résolution insuffisante | seuil IoU, résolution, données de groupes |
| bon résultat sur une vidéo seulement | surapprentissage au décor | séparation par vidéo ou lieu, données variées |
| GPU peu occupé | décodage ou préparation trop lente, batch trop petit | profiler toute la chaîne, ajuster le batch |

---

## 20. Pièges de raisonnement fréquents

### « Une confiance de 90 % signifie que la prédiction est correcte neuf fois sur dix »

Pas nécessairement. Cela ne devient vrai que si le modèle est correctement calibré sur des données comparables au cas d'utilisation. Il faut le vérifier expérimentalement.

### « Si je baisse `conf`, le modèle devient meilleur »

Le modèle ne change pas : on conserve simplement davantage de propositions. Le rappel peut augmenter, mais les faux positifs aussi.

### « Une grande boîte autour de l'objet est forcément correcte »

La classe peut être juste alors que la localisation est mauvaise. L'IoU permet de mesurer cette différence.

### « Le modèle reconnaît le concept comme un humain »

Il exploite des régularités apprises dans les données. Il peut se tromper à cause d'une texture, d'un arrière-plan ou d'une condition jamais rencontrée.

### « Plus le modèle est gros, mieux il détectera »

Pas toujours. Un modèle plus grand coûte davantage et ne compense pas des annotations incohérentes, une résolution insuffisante ou un fort décalage entre entraînement et utilisation.

### « Une vidéo fournit beaucoup d'images indépendantes »

Les trames voisines sont très similaires. Mille trames d'une seule séquence apportent souvent moins de diversité que cent images prises dans des lieux et des conditions variés.

### « Compter les boîtes donne le nombre d'objets »

Dans une image unique, c'est une approximation possible après suppression des doublons. Dans une vidéo, le même objet apparaît sur plusieurs trames : il faut généralement un suivi ou une logique d'événement.

### « Agrandir une image révèle les détails manquants »

Un redimensionnement crée davantage de pixels, mais pas nécessairement davantage d'information réelle. La résolution optique et la taille originale de l'objet restent déterminantes.

---

## 21. Résumé mental à retenir

```text
Image ou trame vidéo
└─ devient un tenseur de nombres
   └─ traverse le réseau YOLO
      ├─ extraction des caractéristiques
      ├─ combinaison de plusieurs échelles
      └─ production de nombreux candidats
         └─ filtrage par confiance et NMS
            └─ résultat final
               ├─ classe
               ├─ score
               └─ rectangle
```

Les huit idées essentielles sont :

1. détecter signifie **classifier et localiser plusieurs objets** ;
2. la résolution d'entrée détermine une partie de l'information accessible ;
3. PyTorch exécute les calculs du réseau, tandis qu'Ultralytics facilite l'utilisation de YOLO et qu'OpenCV manipule les images et vidéos ;
4. YOLO produit rapidement de nombreuses propositions en une passe du réseau ;
5. le seuil de confiance règle le compromis entre objets trouvés et fausses alertes ;
6. l'IoU mesure le chevauchement et la NMS élimine les doublons ;
7. précision, rappel et mAP mesurent des aspects différents de la qualité ;
8. de bonnes données, représentatives et bien séparées, sont indispensables pour généraliser.

La meilleure façon de comprendre un détecteur est de le considérer comme un système complet : caméra, image, prétraitement, modèle, seuils, post-traitement et métriques. Améliorer une seule pièce ne garantit pas que l'ensemble réponde mieux au besoin réel.

---

## Glossaire rapide

| Terme | Définition courte |
|---|---|
| annotation | réponse correcte associée à une image d'entraînement |
| *batch* | groupe d'images traité ensemble |
| *bounding box* | rectangle englobant un objet |
| classe | catégorie prédite, par exemple `bird` |
| confiance | score attribué à une prédiction |
| framework | ensemble d'outils structurant la construction et l'exécution d'un modèle |
| inférence | utilisation d'un modèle entraîné sur de nouvelles données |
| IoU | mesure du chevauchement de deux rectangles |
| NMS | méthode de suppression des boîtes en doublon |
| poids | paramètres appris par le réseau |
| rappel | part des objets réels qui ont été trouvés |
| précision | part des détections qui étaient correctes |
| tenseur | tableau multidimensionnel de nombres |
| *tracking* | association d'un même objet entre plusieurs trames |

