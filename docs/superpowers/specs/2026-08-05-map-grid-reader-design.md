# Map-grid reader (Brique 1) — design

**Date** : 2026-08-05
**Statut** : design validé, prêt pour writing-plans

## Contexte & problème

Objectif global (multi-sessions) : battre le rival route_103 en autonomie depuis
route_101. Bloqueur : la navigation autonome ne franchit pas route_101.

Deux causes prouvées cette session :
1. **`map_map` thrash** — le survey route sur des arêtes non prouvées ; une arête
   physiquement bloquée n'est jamais enregistrée → reproposée à l'infini →
   `budget_exhausted` au lieu de `complete`/`left_map`. C'est ce bug qui a produit
   ma fausse conclusion « route_101 est une boîte fermée ».
2. **exploration ni ledge-aware ni battle-résiliente** — les barrières à sens
   unique (ledges) sont lues comme des murs ; le backtrack DFS est cassé par
   elles ; l'attrition d'équipe provoque des whiteouts.

Fait de domaine (user) : le bloqueur au milieu de route_101 est une **ledge à sens
unique** (on saute vers le bas, jamais on ne remonte) ; le chemin voulu passe à
DROITE au bout de la ledge puis remonte. Ces ledges sont partout dans le jeu.

Racine commune : la nav n'a **aucune vérité-terrain** de la carte. `WallMap`
n'apprend les murs que par collision physique (bump), sans notion de ledge ni de
passabilité dirigée. Décision user : arrêter le probing physique aveugle et
**lire la vraie carte depuis la RAM** pour voir murs / herbe / ledges + leur sens.

## Périmètre — Brique 1 : lecteur seul

Cette brique livre **uniquement un lecteur** de la grille-carte chargée en RAM.
Elle ne recâble aucune navigation.

**Non-goals explicites (= Brique 2, son propre spec)** :
- aucune modif de `WallMap`, `local_navigator`, `map_map`, `map_traveler`
- pas de re-traversée route_101 → Oldale
- pas de routage passabilité-dirigée
- pas de sémantique de contournement des ledges (mapper `LEDGE_*` → stratégie de
  détour). La Brique 1 **classifie** seulement ; l'exploiter = Brique 2.

## Source de vérité

**Lecture RAM live** de la carte chargée (`gBackupMapLayout` / VMap dans
pokeemerald), décodée par tuile. Le moteur BPEF (FR) est identique à pokeemerald ;
seules les adresses sont relocalisées → découverte d'adresses requise (voir plus
bas, c'est le vrai risque).

## Architecture & interface

Nouveau module `env/map_grid_reader.py`, même patron que `EmeraldReader` /
`BattleReader` : une callable `read(addr, size)` injectée, zéro dépendance ROM/SB3.

```python
from __future__ import annotations
from enum import Enum

class TileKind(Enum):
    FREE = 0
    WALL = 1
    GRASS = 2
    LEDGE_UP = 3
    LEDGE_DOWN = 4
    LEDGE_LEFT = 5
    LEDGE_RIGHT = 6

class MapGridReader:
    def __init__(self, read: ReadFn) -> None: ...
    def dimensions(self) -> tuple[int, int] | None       # (width, height) carte chargée
    def tile_behavior_at(self, x: int, y: int) -> int | None
    def classify_at(self, x: int, y: int) -> TileKind | None
    def grid(self) -> list[list[TileKind]] | None          # dump complet classifié
```

`grid()` est fourni pour le tool jetable + les tests + le futur decode-once. Il
n'est **pas** destiné à la boucle chaude (une carte peut être grande — 100×100 =
10k classifications) ; la nav live utilisera `classify_at` ponctuel. À documenter.

### Décodage hybride (par tuile)

1. lire l'entrée `u16` du buffer à l'index de la tuile
2. `collision = (entry & 0x0C00) >> 10` → si ≠0 : **WALL** (stop)
3. sinon `metatile_id = entry & 0x03FF`
4. `metatile_id` → attributs du tileset → **behavior**
5. behavior → `GRASS` / `LEDGE_<dir>` / sinon **FREE**

### Point d'intégration unique

`WorldReader._tile_behavior()` (aujourd'hui stub `return None` avec TODO) appellera
`MapGridReader.tile_behavior_at(px, py)` pour remplir le champ `tile_behavior` déjà
présent dans `SnapshotState`. Rien d'autre dans `WorldReader` ne change.

## Inconnus à résoudre par la sonde de découverte (NE PAS deviner)

Ces points viennent de ma mémoire pokeemerald générique et **doivent être validés
empiriquement** contre BPEF avant tout code figé. Un implémenteur ne doit PAS coder
une formule tirée de cette section sans la confirmer par la sonde.

1. **Formule d'indexation du buffer.** Dans pokeemerald `gBackupMapLayout.width`
   inclut déjà le padding de bordure (`mapWidth + 15`), et les coords joueur
   (`gSaveBlock1.pos`) sont en tuiles-carte, pas buffer. L'offset de bordure
   (`MAP_OFFSET`=7 des deux côtés → +15 sur le stride, +7 sur x et y ?) est à
   confirmer. Formule provisoire : `idx = (x + off) + (y + off) * stride`, où
   `stride` et `off` sont **inconnus tant que la sonde ne les a pas prouvés**.
2. **Largeur du champ behavior.** Les metatile attributes sont un `u16` par
   metatile ; le behavior = bits bas (`attributes & 0x00FF` en Emerald, à
   confirmer par masque). Pas un simple byte isolé.
3. **Chaînage des deux tilesets.** `metatile_id ≥ 0x200` → secondary tileset
   (id − 0x200) ; les pointeurs d'attributs (primary ET secondary) vivent dans
   `gMapHeader.mapLayout.{primary,secondary}Tileset.metatileAttributes`. La sonde
   doit d'abord localiser `gMapHeader`, puis déréférencer 2 pointeurs. Le
   « deref_tileset_attr » de l'interface cache ces 2 indirections.

Adresses à trouver sur BPEF : `gBackupMapLayout` (ptr buffer), largeur/hauteur,
`gMapHeader` + tables d'attributs des 2 tilesets, `MAP_OFFSET`.

**Méthode de découverte** : sonde jetable qui balaye la RAM et valide contre la
géométrie CONNUE de `post_starter` (joueur en (10,17) ; ledge au milieu ; bande
d'herbe à l'ouest, colonne x≈2). Une adresse/formule n'est retenue que si la grille
décodée reproduit ces faits. **Si la découverte échoue en budget raisonnable →
remonter à l'user avant d'écrire du code spéculatif.**

## Gestion d'erreurs (valider à la frontière RAM)

- pointeur/dimensions nuls ou aberrants (warp, incohérence SaveBlock1 1-tick déjà
  connue, buffer partiellement réécrit en transition) → rendre `None`. Contrat
  best-effort : un snapshot peut être périmé d'1 tick (déjà le contrat WorldReader).
- dimensions bornées plausibles (ex. w,h ∈ [1, 256]) sinon `None`
- `metatile_id == 0x3FF` (marqueur corruption, réf heatz123) → `WALL` (sûr)
- (x,y) hors bornes → `None`

## Tests

**Purs (fake `read` sur buffer forgé, zéro ROM)** :
- `dimensions()` lit w/h
- `collision != 0` → `WALL`
- behavior herbe → `GRASS`
- behavior ledge est → `LEDGE_RIGHT`
- split `metatile_id` (masque `0x03FF`) correct
- frontière tileset primary/secondary à `0x200`
- marqueur `0x3FF` → `WALL`
- (x,y) OOB → `None`

**Smoke ROM gaté** (`states/post_starter.state`, skip si ROM/état absent) :
- `dimensions()` plausibles
- tuile joueur (10,17) = `FREE`
- **cross-check** : au moins une tuile **mur franc** (arbre / bord de carte),
  confirmée bloquée par un bump-test WallMap live, ressort `WALL` dans la grille.
  **Choisir un mur franc, PAS une ledge** : une ledge est collision=0 (grille dit
  `LEDGE_*`) mais bump-bloquée en montée → assert `WALL` échouerait à tort.

## Critère de validation

La grille décodée de route_101 depuis `post_starter` reproduit les 3 faits connus
(joueur (10,17) FREE, ledge au milieu classée `LEDGE_*`, herbe ouest `GRASS`) ET
le tool jetable rend une vue lisible où la ledge à sens unique et la connexion nord
vers Oldale sont visibles.

## Livrable annexe

`tools/dump_map_grid.py` (jetable) — charge `post_starter`, rend la grille route_101
classifiée en ASCII/couleur → répond visuellement à « où sont les problèmes » (voir
la ledge + la connexion nord Oldale).

## Suite (hors périmètre)

Brique 2 (son propre spec) : recâbler `WallMap` / `local_navigator` / `map_map`
pour consommer cette vérité-terrain (ledge-aware + fin du thrash) + re-traverser
route_101 → Oldale → route_103.
