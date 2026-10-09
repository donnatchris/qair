# Comprendre le GPU et le développement sur carte graphique

Ce document présente les bases nécessaires pour comprendre ce qu'est un GPU, comment un programme l'utilise et comment interpréter les mesures affichées par `nvtop`.

> **Idée centrale :** un CPU cherche surtout à exécuter rapidement quelques suites d'instructions complexes. Un GPU cherche surtout à appliquer la même opération à une très grande quantité de données en parallèle.

---

## 1. Le GPU en une analogie

Imaginons deux cuisines :

- le **CPU** est une petite brigade composée de quelques chefs très polyvalents. Chaque chef sait improviser, prendre des décisions complexes et changer rapidement de tâche ;
- le **GPU** est une immense brigade composée de centaines ou de milliers d'exécutants. Chacun est moins autonome, mais tous peuvent répéter simultanément la même recette sur des ingrédients différents.

Pour préparer quatre plats tous différents, les chefs du CPU sont souvent plus efficaces. Pour appliquer la même découpe à un million de légumes, la brigade du GPU est généralement beaucoup plus rapide.

Cette différence explique pourquoi les GPU sont efficaces pour :

- le rendu 2D et 3D ;
- le traitement d'images et de vidéos ;
- l'apprentissage et l'inférence de réseaux de neurones ;
- les simulations scientifiques ;
- les calculs matriciels et vectoriels massivement parallèles.

Un GPU n'est donc pas « un CPU plus rapide ». C'est un processeur conçu pour un autre type de travail.

---

## 2. CPU et GPU : deux architectures complémentaires

| Caractéristique | CPU | GPU |
|---|---|---|
| Nombre d'unités de calcul | Faible à modéré | Très élevé |
| Force principale | Faible latence, logique complexe | Débit massif, parallélisme |
| Contrôle de flux | Très performant | Coûteux lorsque les chemins divergent |
| Cache par unité de calcul | Important | Plus réduit, partagé entre beaucoup de threads |
| Tâche idéale | Suite d'étapes dépendantes | Même calcul sur beaucoup de données |
| Exemple | Piloter l'application | Multiplier de grandes matrices |

### Latence et débit

Deux notions permettent de résumer la différence :

- la **latence** est le temps nécessaire pour terminer une tâche donnée ;
- le **débit** est la quantité totale de travail terminée par unité de temps.

Le CPU est optimisé pour réduire la latence d'une tâche. Le GPU accepte qu'une tâche individuelle ne soit pas instantanée, mais en garde un très grand nombre en vol afin d'obtenir un débit élevé.

Analogie : une voiture de sport transporte rapidement quelques personnes ; un train transporte beaucoup plus de personnes à la fois. Selon le besoin, « le plus rapide » n'a pas le même sens.

---

## 3. Les principaux composants d'une carte graphique

Une carte graphique dédiée ne contient pas seulement un GPU.

```text
Programme sur le CPU
        │
        │ commandes et données, généralement via PCIe
        ▼
┌──────────────── Carte graphique ────────────────┐
│  GPU                                            │
│  ├─ unités de calcul                            │
│  ├─ unités spécialisées                         │
│  └─ caches                                      │
│                                                 │
│  VRAM ─ contrôleurs mémoire ─ bus mémoire       │
│                                                 │
│  alimentation ─ capteurs ─ refroidissement      │
└─────────────────────────────────────────────────┘
```

### 3.1 Les unités de calcul

Le GPU regroupe ses unités de calcul en ensembles. Leur nom dépend du constructeur :

- **SM** (*Streaming Multiprocessor*) chez NVIDIA ;
- **CU** (*Compute Unit*) chez AMD ;
- **Xe-core** chez Intel.

Ces ensembles exécutent de nombreux threads simultanément. Ils contiennent notamment des unités arithmétiques, des registres, une mémoire locale rapide et des ordonnanceurs.

### 3.2 Les unités spécialisées

Selon le modèle, le GPU peut aussi posséder :

- des unités matricielles, par exemple les *Tensor Cores*, utiles à l'IA ;
- des unités de lancer de rayons pour le rendu 3D ;
- des moteurs dédiés à l'encodage et au décodage vidéo ;
- des moteurs de copie capables de déplacer des données en parallèle du calcul.

Une activité importante du décodeur vidéo ne signifie donc pas nécessairement que les unités de calcul générales sont occupées.

### 3.3 La VRAM

La **VRAM** est la mémoire directement accessible par le GPU. Elle contient par exemple :

- les poids d'un modèle de machine learning ;
- les tenseurs intermédiaires ;
- les images et textures ;
- les tampons d'entrée et de sortie ;
- le code des noyaux de calcul et les espaces de travail des bibliothèques.

Elle offre une très grande bande passante, mais sa capacité est limitée. Une carte de 8 Gio ne peut pas conserver un ensemble de données de 10 Gio en VRAM sans le découper, le compresser ou en déporter une partie.

> **Attention :** avoir assez de VRAM évite une erreur de mémoire, mais ne garantit pas que le GPU calcule vite. La capacité, la bande passante et la puissance de calcul sont trois propriétés différentes.

### 3.4 Le lien PCI Express

Sur une machine classique, le CPU et la carte graphique communiquent par le bus **PCI Express**, ou PCIe. Ce lien est généralement beaucoup plus lent que la mémoire interne du GPU.

Analogie : la VRAM est l'entrepôt situé dans l'usine ; la RAM centrale est un entrepôt situé de l'autre côté d'une route. Même si les deux entrepôts sont rapides, les allers-retours sur la route coûtent du temps.

### 3.5 Le cas des GPU intégrés

Un GPU intégré peut partager la mémoire principale avec le CPU. La frontière entre RAM et VRAM devient alors moins nette. Cela réduit certains transferts explicites, mais CPU et GPU se partagent la capacité et la bande passante mémoire.

---

## 4. Le modèle d'exécution : beaucoup de threads regroupés

### 4.1 Le noyau de calcul, ou *kernel*

En programmation GPU, un **kernel** est une fonction exécutée en parallèle par un grand nombre de threads.

Exemple conceptuel d'addition de deux tableaux :

```c
// Pseudo-code : chaque thread traite un indice différent.
kernel addition(a, b, resultat, i) {
    resultat[i] = a[i] + b[i];
}
```

Au lieu d'écrire une boucle séquentielle qui parcourt tous les éléments, le programme lance potentiellement un thread par élément.

### 4.2 Grilles, blocs et groupes de threads

Les API organisent généralement le travail de façon hiérarchique :

```text
Grille de travail
├─ bloc 0
│  ├─ thread 0
│  ├─ thread 1
│  └─ ...
├─ bloc 1
└─ ...
```

Les threads d'un même bloc peuvent habituellement coopérer grâce à une mémoire partagée rapide et à des mécanismes de synchronisation.

Le matériel exécute les threads par petits groupes :

- un **warp** chez NVIDIA, généralement 32 threads ;
- un **wavefront** chez AMD, dont la largeur dépend de l'architecture et du mode utilisé ;
- un **subgroup** dans une terminologie plus générique.

Ce modèle est souvent appelé **SIMT** : *Single Instruction, Multiple Threads*. Un même groupe applique en grande partie la même instruction à plusieurs données.

### 4.3 La divergence

Considérons ce code :

```c
if (valeur > 0) {
    calcul_A();
} else {
    calcul_B();
}
```

Si la moitié des threads d'un groupe choisit `calcul_A` et l'autre moitié `calcul_B`, le GPU peut devoir exécuter successivement les deux branches en masquant les threads non concernés. C'est la **divergence**.

Analogie : si une classe entière suit la même consigne, tout le monde avance ensemble. Si chaque élève demande une consigne différente, le professeur doit traiter les cas séparément.

Une condition n'est pas automatiquement mauvaise. Elle devient coûteuse surtout lorsque les threads d'un même groupe suivent des chemins différents.

### 4.4 Masquer les attentes

Un accès mémoire peut prendre du temps. Pour éviter de rester inactif, le GPU met en pause un groupe de threads en attente et en exécute un autre déjà prêt.

Il faut donc généralement lancer assez de threads pour occuper le matériel et masquer les latences. Cette capacité d'occupation est liée à l'**occupancy**, mais une occupancy maximale ne garantit pas à elle seule les meilleures performances.

---

## 5. La hiérarchie mémoire

Toutes les mémoires ne se valent pas :

| Niveau | Portée typique | Vitesse relative | Capacité relative |
|---|---|---:|---:|
| Registres | Un thread | Très élevée | Minuscule |
| Mémoire partagée / cache local | Un bloc ou une unité de calcul | Élevée | Petite |
| Caches | Plusieurs unités | Élevée à moyenne | Petite à moyenne |
| VRAM / mémoire globale | Tout le GPU | Plus lente | Grande |
| RAM de la machine via PCIe | CPU et transferts | Bien plus lente à atteindre | Très grande |

L'objectif n'est pas d'éviter la VRAM : elle est indispensable. Il faut surtout :

- réutiliser les données déjà chargées ;
- effectuer des accès voisins et réguliers ;
- éviter les transferts CPU ↔ GPU inutiles ;
- limiter les allocations et libérations répétées ;
- regrouper le travail en lots suffisamment grands.

### Accès coalescés

Lorsque des threads voisins lisent des adresses voisines, le GPU peut regrouper les accès mémoire. On parle souvent d'accès **coalescés**.

Analogie : il est plus efficace pour un livreur de déposer dix colis dans la même rue que de traverser la ville entre chaque colis.

---

## 6. Comment une application utilise le GPU

Une exécution simplifiée suit souvent ce parcours :

```text
1. Le CPU prépare les données
          ↓
2. Les données sont placées en VRAM
          ↓
3. Le CPU demande le lancement d'un kernel
          ↓
4. Le GPU exécute de nombreux threads
          ↓
5. Les résultats restent en VRAM ou reviennent en RAM
          ↓
6. Le CPU poursuit le programme
```

Le CPU est appelé **hôte** (*host*) et le GPU **périphérique** (*device*).

### 6.1 Les couches logicielles

Une application n'envoie pas directement des signaux électriques à la carte. Elle passe par plusieurs couches :

```text
Application Python ou C++
        ↓
Framework ou bibliothèque : PyTorch, TensorFlow, OpenCV, BLAS…
        ↓
API et runtime : CUDA, ROCm/HIP, OpenCL, Vulkan, oneAPI…
        ↓
Pilote du constructeur
        ↓
GPU
```

Dans un programme Python, la boucle de haut niveau reste pilotée par le CPU. Les opérations lourdes appellent généralement des kernels compilés en C/C++ et en code GPU.

### 6.2 Exécution asynchrone

Le lancement d'un kernel est souvent **asynchrone** : le CPU dépose une commande dans une file puis continue sans attendre immédiatement sa fin.

Conséquences :

- chronométrer uniquement l'appel Python peut sous-estimer le temps réel ;
- demander un résultat au CPU force souvent une synchronisation ;
- plusieurs transferts et calculs peuvent parfois se chevaucher grâce à des *streams* ou files de commandes.

Pour mesurer correctement une opération, il faut utiliser les événements fournis par l'API GPU ou synchroniser explicitement autour du chronométrage.

### 6.3 Exemple avec un tenseur

Dans un framework de machine learning, le changement de périphérique est conceptuellement le suivant :

```python
# Exemple conceptuel, avec une API proche de PyTorch.
x = creer_tenseur_sur_cpu()
x_gpu = x.vers_le_gpu()          # transfert RAM -> VRAM
y_gpu = modele(x_gpu)            # kernels exécutés sur le GPU
y = y_gpu.vers_le_cpu()          # transfert VRAM -> RAM
```

Si chaque petite opération provoque un aller-retour, le coût des transferts peut dépasser celui du calcul. Il est préférable de garder les données et les opérations liées sur le GPU aussi longtemps que possible.

---

## 7. Ce qui limite les performances

Un programme GPU peut être limité par plusieurs ressources.

### 7.1 Limité par le calcul (*compute-bound*)

Les unités de calcul sont très occupées et exécutent beaucoup d'opérations arithmétiques. Exemples : grandes multiplications de matrices, convolutions denses.

Signes possibles :

- utilisation GPU élevée et stable ;
- fréquence GPU élevée ;
- consommation électrique importante ;
- les changements de précision, comme FP32 vers FP16, peuvent beaucoup aider.

### 7.2 Limité par la mémoire (*memory-bound*)

Le GPU passe beaucoup de temps à déplacer des données depuis la VRAM. Ajouter des unités arithmétiques n'aiderait pas nécessairement.

Signes possibles :

- utilisation GPU significative, mais performances sensibles à la disposition des données ;
- amélioration lorsque les accès deviennent contigus ou lorsque les données sont réutilisées en cache ;
- la fréquence mémoire peut jouer un rôle important.

`nvtop` ne montre pas toujours directement la bande passante interne de la VRAM. Une utilisation mémoire en pourcentage ne doit pas être confondue avec la quantité de VRAM allouée.

### 7.3 Limité par les transferts

Le GPU attend les données provenant du CPU ou renvoie très souvent de petits résultats.

Signes possibles :

- activité PCIe RX/TX importante ;
- utilisation GPU en dents de scie ;
- nombreuses petites opérations séparées par des synchronisations.

### 7.4 Limité par le CPU ou les entrées-sorties

Le GPU est prêt, mais le CPU, le disque, le décodage ou le chargement des données ne le nourrit pas assez vite.

Exemple en entraînement : le GPU termine un lot puis attend le chargement et l'augmentation des images du lot suivant.

Signes possibles :

- VRAM allouée, mais utilisation GPU basse ou périodique ;
- un cœur CPU saturé ;
- accélération lorsque le nombre de workers de chargement augmente, jusqu'à une certaine limite.

### 7.5 Limité par la capacité de VRAM

Quand la VRAM est presque pleine, plusieurs cas sont possibles :

- tout fonctionne normalement : la mémoire est simplement utilisée efficacement ;
- une nouvelle allocation échoue avec une erreur de type *out of memory* ;
- le système utilise une mémoire partagée ou des migrations de pages, avec une forte baisse de performances selon la plateforme.

Une VRAM occupée à 95 % n'implique donc pas que le GPU « travaille à 95 % ».

---

## 8. Comprendre `nvtop`

`nvtop`, pour **Neat Videocard TOP**, est un moniteur interactif comparable à `top` ou `htop`, mais destiné aux GPU et accélérateurs. Son écran comporte généralement :

1. un résumé instantané de chaque GPU ;
2. des graphiques montrant l'évolution dans le temps ;
3. une liste des processus utilisant les GPU.

L'interface exacte et les métriques disponibles dépendent de la version de `nvtop`, du constructeur, du modèle, du pilote, du noyau Linux et des permissions. Un champ `N/A` signifie souvent que l'information n'est pas exposée, pas que la valeur réelle est nulle.

### 8.1 Résumé du périphérique

| Indicateur courant | Signification | Comment l'interpréter |
|---|---|---|
| `GPU` / `GPU-Util` | Part du temps récent pendant laquelle un ou plusieurs moteurs de calcul étaient actifs | 100 % indique une forte occupation pendant la fenêtre de mesure, pas nécessairement 100 % du débit théorique |
| `MEM` / `MEM-Util` | Taux d'utilisation mémoire communiqué ou reconstruit par le pilote | Il représente généralement l'activité du contrôleur mémoire ; sur certains GPU à mémoire partagée, `nvtop` peut le déduire de la mémoire occupée. Il faut donc vérifier le backend utilisé avant de l'assimiler à une bande passante |
| mémoire utilisée / totale | Capacité de VRAM actuellement allouée ou comptabilisée | Par exemple `6.2 GiB / 8 GiB` décrit une quantité, pas une vitesse |
| `TEMP` | Température d'un capteur du GPU, en °C par défaut | Une hausse sous charge est normale ; une température proche d'une limite peut provoquer une réduction de fréquence |
| `FAN` | Vitesse du ventilateur, en pourcentage ou en tr/min selon le matériel | `0 %` peut être normal sur une carte possédant un mode semi-passif |
| `POW` / `PWR` | Puissance électrique instantanée, souvent en watts | À comparer à la limite de puissance lorsque celle-ci est affichée |
| fréquence GPU | Fréquence actuelle du cœur ou domaine de calcul, en MHz | Elle varie dynamiquement avec la charge, la température et les limites électriques |
| fréquence mémoire | Fréquence actuelle de la mémoire vidéo, en MHz | Sa représentation varie selon le constructeur ; ne pas comparer aveuglément deux architectures |
| `ENC` | Utilisation du moteur matériel d'encodage vidéo | Peut être élevée alors que le calcul général est faible |
| `DEC` | Utilisation du moteur matériel de décodage vidéo | Utile pour diagnostiquer lecture vidéo ou pipeline caméra |
| PCIe `RX` | Données reçues par la carte via PCIe | Correspond typiquement à des transferts hôte vers périphérique, mais le sens exact est celui vu par le GPU |
| PCIe `TX` | Données émises par la carte via PCIe | Correspond typiquement à des transferts périphérique vers hôte |
| lien PCIe | Génération et largeur actives, par exemple Gen4 ×16 | Le lien peut réduire sa vitesse au repos pour économiser l'énergie |
| `Effective load` | Produit de l'utilisation GPU par le rapport `puissance actuelle / puissance maximale` | Cette estimation pondère l'occupation par la puissance consommée ; elle n'est disponible que si le pilote expose les trois valeurs nécessaires et ne mesure pas directement les performances applicatives |

#### Trois pourcentages à ne pas confondre

```text
GPU 95 %          unités de calcul actives presque tout le temps
MEM 70 %          sous-système mémoire très actif
VRAM 7 / 8 Gio    quantité de mémoire réservée ou utilisée
```

Ils répondent à trois questions différentes :

1. le GPU calcule-t-il ?
2. la mémoire transfère-t-elle beaucoup de données ?
3. combien de données résident actuellement en VRAM ?

### 8.2 Les graphiques

La zone centrale montre l'historique récent. Selon la configuration, elle peut tracer :

- l'utilisation GPU ;
- l'utilisation de la mémoire ;
- l'encodage et le décodage ;
- la température ;
- la puissance ;
- la vitesse du ventilateur ;
- les fréquences du GPU et de sa mémoire ;
- la charge effective ;
- les charges PCIe RX et TX.

Pour PCIe, `nvtop` peut exprimer RX/TX en pourcentage de la bande passante maximale du lien actif, déterminée notamment par sa génération et sa largeur.

Un graphique est souvent plus informatif que la valeur instantanée :

- un plateau élevé correspond à une charge continue ;
- des pics séparés peuvent signaler de petits lots ou des attentes ;
- une baisse de fréquence accompagnée d'une température élevée peut indiquer une limitation thermique ;
- une alternance calcul/transfert peut révéler un pipeline insuffisamment chevauché.

Appuyer sur `F2`, puis ouvrir la section **Chart**, permet de choisir les métriques tracées.

### 8.3 La liste des processus

Les colonnes disponibles varient, mais on rencontre couramment :

| Colonne | Signification |
|---|---|
| `PID` | Identifiant du processus système |
| `USER` | Utilisateur propriétaire du processus |
| `GPU` ou `GPU ID` | GPU utilisé sur une machine qui en possède plusieurs |
| `TYPE` | Nature de l'activité, par exemple graphique, calcul ou les deux selon le pilote |
| `GPU%` | Utilisation du GPU attribuée au processus pendant la fenêtre de mesure |
| `ENC%` | Utilisation de l'encodeur vidéo attribuée au processus |
| `DEC%` | Utilisation du décodeur vidéo attribuée au processus |
| `MEM` ou `GPU MEM` | Quantité de mémoire GPU attribuée au processus |
| `CPU%` | Utilisation CPU du processus ; 100 % représente souvent environ un cœur logique entièrement occupé sous Linux |
| `CPU MEM` | Mémoire centrale utilisée par le processus |
| `COMMAND` | Nom ou ligne de commande du processus |

Quelques précautions :

- la somme des mémoires par processus ne correspond pas toujours exactement au total, car le pilote, le contexte graphique, les allocations partagées et les tampons internes consomment aussi de la mémoire ;
- l'attribution de l'utilisation GPU à chaque processus dépend des capacités du pilote ;
- un processus peut conserver beaucoup de VRAM tout en étant momentanément inactif ;
- un navigateur ou le serveur d'affichage peut légitimement apparaître dans la liste ;
- tuer un processus avec `F9` lui fait perdre son travail non sauvegardé.

### 8.4 Température, puissance et fréquences : la même histoire

Ces mesures doivent être lues ensemble :

```text
charge élevée
    ↓
fréquence élevée
    ↓
puissance et température augmentent
    ↓
ventilateur accélère
```

Si une limite thermique ou électrique est atteinte, la carte peut réduire sa fréquence : c'est le **throttling**. Une fréquence plus basse n'est toutefois pas une preuve suffisante de throttling : le GPU baisse aussi normalement sa fréquence lorsqu'il a peu de travail.

### 8.5 Raccourcis utiles

| Touche | Action |
|---|---|
| `F2` | Ouvrir les réglages de l'interface |
| `F6` | Choisir la colonne de tri des processus |
| `+` / `-` | Trier dans l'ordre croissant / décroissant |
| `↑` / `↓` | Sélectionner un processus |
| `←` / `→` | Faire défiler une ligne de processus |
| `F9` | Choisir un signal à envoyer au processus sélectionné |
| `F12` | Enregistrer les préférences |
| `F10`, `q` ou `Échap` | Quitter |

Commandes pratiques :

```bash
nvtop --help          # afficher les options disponibles dans la version installée
man nvtop             # consulter le manuel local
nvtop -s              # produire un instantané sans interface interactive
nvtop -l              # produire des instantanés successifs
nvtop -d 20           # rafraîchir toutes les 2 secondes (20 × 0,1 s)
nvtop -P              # masquer la liste des processus
nvtop -i              # demander les informations GPU dynamiques supplémentaires
nvtop -X              # demander aussi les caractéristiques statiques supplémentaires
```

Toutes ces options ne sont pas nécessairement présentes dans une ancienne version. `nvtop --help` reste la référence pour l'installation locale.

---

## 9. Lire quelques scénarios typiques dans `nvtop`

### Scénario A — le GPU travaille correctement

```text
GPU : 90–100 %    VRAM : stable    puissance : élevée    PCIe : modéré
```

Le GPU reçoit assez de travail et calcule presque continuellement. Il reste à vérifier que le temps obtenu répond au besoin ; une utilisation élevée n'est pas un objectif en soi.

### Scénario B — beaucoup de mémoire, peu de calcul

```text
GPU : 0–10 %      VRAM : 90 %      processus présent
```

Le programme conserve probablement un modèle ou des données en VRAM, mais ne lance presque aucun calcul à cet instant. Il peut attendre le CPU, le disque, le réseau ou une synchronisation.

### Scénario C — utilisation en dents de scie

```text
GPU : 100 %, puis 0 %, puis 100 %, puis 0 %…
```

Causes possibles :

- préparation des lots trop lente sur le CPU ;
- opérations GPU trop petites ;
- synchronisations fréquentes ;
- lecture disque ou décodage irrégulier ;
- inférence déclenchée seulement à l'arrivée de nouvelles données.

### Scénario D — activité PCIe très forte

```text
PCIe RX/TX : élevé    GPU : moyen ou irrégulier
```

Le programme déplace peut-être trop souvent ses données entre RAM et VRAM. Il faut rechercher des conversions vers le CPU, des copies implicites ou des lots trop petits.

### Scénario E — température haute et fréquence qui chute

```text
GPU : élevé    TEMP : élevée    FAN : élevé    fréquence : baisse progressive
```

Une limitation thermique est possible. Il faut comparer la température aux spécifications du modèle, vérifier la ventilation et utiliser un outil constructeur pour confirmer la raison de limitation.

### Scénario F — `ENC` ou `DEC` élevé, `GPU%` modéré

Un moteur vidéo spécialisé travaille. C'est normal lors de l'encodage d'un flux, de la lecture d'une vidéo ou de certains pipelines caméra. Les moteurs vidéo et les unités de calcul générales sont distincts.

---

## 10. Une méthode simple de diagnostic

Lorsqu'un programme GPU semble lent :

1. **Vérifier qu'il utilise le bon GPU.** Repérer son PID et son identifiant GPU dans `nvtop`.
2. **Observer une durée suffisante.** Ne pas conclure à partir d'une seule valeur instantanée.
3. **Distinguer capacité et activité mémoire.** VRAM occupée et bande passante utilisée sont deux mesures différentes.
4. **Regarder le rythme.** Une charge stable et une charge en pics racontent des histoires différentes.
5. **Comparer GPU, CPU et PCIe.** Un GPU peu actif peut attendre une autre partie du système.
6. **Vérifier température, puissance et fréquence ensemble.** Cela aide à détecter une limite thermique ou électrique.
7. **Mesurer dans l'application.** `nvtop` donne des indices ; un profileur identifie précisément les kernels, copies et attentes responsables.

Outils de profilage courants :

- NVIDIA Nsight Systems et Nsight Compute ;
- AMD ROCm Profiler et Omniperf ;
- Intel VTune et les outils oneAPI ;
- les profileurs intégrés à PyTorch ou TensorFlow.

`nvtop` est l'équivalent d'un tableau de bord automobile : il montre vitesse, température et régime. Pour savoir exactement quelle pièce ralentit le moteur, il faut ensuite un outil de diagnostic plus précis.

---

## 11. Pièges de raisonnement fréquents

### « Mon GPU est à 100 %, il délivre donc 100 % de sa puissance théorique »

Non. Le compteur indique surtout que le moteur est occupé pendant la période observée. Il peut être occupé par un code peu efficace, attendre souvent la mémoire ou utiliser seulement certaines unités.

### « Toute la VRAM est occupée, donc le GPU est saturé »

Non. La VRAM mesure un espace de stockage. Le taux d'utilisation GPU mesure une activité. Une bibliothèque peut aussi garder un cache d'allocations pour les réutiliser plus tard.

### « Une température élevée signifie que la carte est en danger »

Pas nécessairement. Les GPU sont conçus pour fonctionner à température élevée et disposent de protections. Il faut comparer la mesure aux limites du modèle et rechercher une baisse de fréquence ou un indicateur de throttling.

### « Un petit calcul sera toujours plus rapide sur GPU »

Non. Le lancement du kernel et le transfert des données ont un coût fixe. Pour une petite quantité de travail, le CPU peut terminer avant même que le GPU soit pleinement lancé.

### « Plus de threads signifie toujours plus de vitesse »

Non. Il faut assez de parallélisme pour occuper la carte, mais les registres, la mémoire partagée, la bande passante et les dépendances deviennent ensuite les facteurs limitants.

---

## 12. Résumé mental à retenir

```text
CPU
└─ organise le programme et prépare le travail
   └─ envoie commandes et données
      └─ GPU
         ├─ exécute beaucoup de threads par groupes
         ├─ est efficace si ces threads font un travail similaire
         ├─ lit et écrit principalement en VRAM
         └─ renvoie éventuellement les résultats au CPU
```

Les cinq idées essentielles sont :

1. un GPU privilégie le **débit parallèle** plutôt que la faible latence ;
2. les données doivent être découpées en nombreux travaux similaires ;
3. les mouvements de données coûtent souvent aussi cher que les calculs ;
4. VRAM occupée, activité mémoire et activité de calcul sont des métriques distinctes ;
5. `nvtop` permet d'observer des symptômes, tandis qu'un profileur permet d'identifier précisément leur cause.

---

## Références

- [Dépôt officiel de `nvtop`](https://github.com/Syllo/nvtop)
- [Manuel officiel de `nvtop`](https://github.com/Syllo/nvtop/blob/master/manpage/nvtop.in)

