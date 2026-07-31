# Interface d'ordres Strategist — design (petit bout)

**Date :** 2026-07-31
**Palier :** P4 étape 1
**Statut :** proposé

---

## En une phrase

Créer le **langage commun** entre les trois cerveaux : un objet `Order`
(« va à tel endroit ») que le Strategist pourra émettre et que l'Explorer sait
exécuter en marchant jusqu'à la destination.

## Pourquoi maintenant

Les trois cerveaux existent mais **ne se parlent pas encore** :

- Le **Strategist** décide (avancer / grinder / soigner) dans un monde abstrait,
  sans carte ni émulateur (`env/strategist_env.py`).
- L'**Explorer** sait vraiment marcher dans le jeu : `navigate_to` (aller à une
  case), `travel_to` (enchaîner les portes de carte en carte), `survey_world`
  (cartographier).
- Le **Fighter** sait gagner un combat.

Il manque la pièce du milieu : un **ordre** que le chef donne et que l'ouvrier
comprend. C'est cette pièce, et **rien d'autre**, que ce petit bout construit.

## Ce qu'on construit (et rien de plus)

Un **seul fichier neuf** : `env/orders.py`. On ne touche à aucun fichier
existant.

### 1. L'objet `Order`

Un ordre figé (immutable), trois champs :

```python
@dataclass(frozen=True)
class Order:
    destination: str   # nom d'un lieu, ex. "route_101"
    mode: str          # "advance" | "grind" | "heal"
    combat: str        # "win" | "capture" | "min_loss"
```

- `destination` : le **nom** d'un endroit (pas des coordonnées). C'est le chef
  qui parle en noms de lieux, comme un humain (« va à la Route 101 »).
- `mode` : ce qu'on fait là-bas. Pour ce petit bout, **seul `"advance"` fait
  vraiment quelque chose** (= s'y rendre). `"grind"` et `"heal"` sont acceptés
  mais renverront `"not_implemented"` — on les câblera plus tard.
- `combat` : la directive de combat, **stockée pour plus tard** (le Fighter n'est
  pas branché ici). On la garde dans l'objet pour que le langage soit complet dès
  maintenant.

### 2. Le carnet d'adresses `DESTINATIONS`

Une table écrite **à la main** qui traduit un nom en (carte, case) :

```python
DESTINATIONS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "littleroot": ((0, 9), (3, 10)),   # Bourg-en-Vol, case d'arrivée du camion
    "route_101":  ((0, 16), (5, 12)),  # Route 101, entrée sud depuis Bourg-en-Vol
}
```

Les cases exactes sont des points de repère connus (Bourg-en-Vol (3,10) = où le
perso atterrit en sortant du camion ; Route 101 (5,12) = case d'entrée sud). Si
une case s'avère fausse à l'implémentation, on la corrige dans la table — c'est
justement l'avantage d'un carnet d'adresses écrit à la main.

**Choix de design (important) :** table écrite à la main, PAS une recherche dans
la mémoire de carte de l'Explorer.

- *Pourquoi ?* C'est le plus simple, et ça **marche même avant toute
  exploration**. Le nom « route_101 » a un sens pour le chef dès le premier pas
  de jeu, avant que l'Explorer ait cartographié quoi que ce soit.
- *Compromis assumé :* si un lieu n'est pas dans la table, on ne peut pas y aller
  (on renvoie `"unknown_destination"`). Brancher la table sur la mémoire de carte
  découverte est un travail **futur**, hors de ce petit bout.

### 3. La fonction `execute_order`

Le pont : elle reçoit un ordre et le fait exécuter par l'Explorer.

```python
def execute_order(
    order: Order,
    emulator, reader, memory, wallmap,
    max_hops: int = 20,
) -> str:
    ...
```

Logique, dans l'ordre :

1. Le nom `order.destination` n'est pas dans `DESTINATIONS`
   → renvoyer `"unknown_destination"`.
2. `order.mode != "advance"` (donc `"grind"` ou `"heal"`)
   → renvoyer `"not_implemented"` (stub, câblage futur).
3. Sinon : résoudre le nom → `(goal_map, goal_cell)` et **déléguer à
   `travel_to`**, en renvoyant tel quel son résultat
   (`arrived` / `unknown_route` / `unreachable` / `lost` / `timeout`).

`execute_order` **n'ajoute aucune logique de navigation** : elle traduit un nom
en coordonnées et passe la main à la brique existante `travel_to`. C'est
volontairement une pièce mince.

## Contrat de `travel_to` (existant, non modifié)

Pour mémoire, la brique appelée :

```python
travel_to(emulator, reader, memory, wallmap, goal_map, goal_cell, max_hops=20) -> str
# "arrived" | "unknown_route" | "unreachable" | "lost" | "timeout"
```

Elle marche de porte en porte **en territoire connu uniquement** ; une porte pas
encore découverte donne `"unknown_route"` (elle n'explore pas — la cartographie
est un autre outil).

## Tests (purs, sans ROM)

On réutilise le fake `WorldGrid` déjà en place dans
`tests/test_world_surveyor.py` (il joue à la fois l'émulateur et le reader). Cas :

1. **Destination connue, même carte** → `execute_order` renvoie `"arrived"`.
2. **Destination connue, plusieurs cartes** (chaîne de portes déjà mémorisées)
   → `"arrived"`.
3. **Nom inconnu** (`"atlantide"`) → `"unknown_destination"`.
4. **Mode non-nav** (`mode="grind"`) → `"not_implemented"`.
5. **Destination connue mais injoignable** (porte pas découverte) →
   `"unknown_route"` (le résultat de `travel_to` remonte tel quel).

Un fichier `tests/test_orders.py`, aucun test ROM (pas de nouveau savestate
nécessaire — `travel_to` a déjà son smoke ROM).

## Hors périmètre (explicitement différé)

- **Ré-entraîner le Strategist** ou changer son MDP.
- **Sortir le scénario** de `env/milestones.py`.
- **Câbler `grind` / `heal`** (renvoient `"not_implemented"` pour l'instant).
- **Brancher le Fighter** sur la directive `combat` (juste stockée).
- **Remplir `DESTINATIONS` depuis la mémoire de carte** (table à la main).
- **Toucher au reward** de qui que ce soit.

## Ce que ce petit bout débloque ensuite

Une fois le langage `Order` posé, les paliers suivants pourront, un par un :
câbler `heal` (aller au Centre Pokémon), câbler `grind` (aller dans l'herbe +
Fighter), brancher le Fighter sur `combat`, et enfin faire émettre de vrais
`Order` par le Strategist. Chacun = son propre petit bout.
