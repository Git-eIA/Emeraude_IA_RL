# Heal mode — reconnaître un lieu de soin et s'y soigner (P4 étape 2)

**Date :** 2026-07-31
**Statut :** design validé, prêt pour le plan

## But (une phrase)

Donner à l'Explorateur sa première compétence de **savoir-reconnaître** : apprendre
tout seul où sont les lieux de soin (en observant que ses PV remontent au max), puis
exécuter l'ordre `mode="heal"` du Stratège de bout en bout (y aller + se soigner).

## Contexte

- L'interface d'ordres (`env/orders.py`, P4 étape 1) existe déjà. `execute_order`
  résout un ordre `mode="advance"` en déléguant à `travel_to`. Les modes `"grind"`
  et `"heal"` renvoient aujourd'hui `"not_implemented"`. **Ce bout remplit `"heal"`.**
- Le Stratège donne des **intentions pures** (choix « A » de la réflexion archi) :
  il dit « soigne-toi », jamais « va au centre de Bourg-en-Vol ». C'est donc à
  l'Explorateur de **savoir quel lieu est un lieu de soin**.
- Fondation « reconnaître par l'effet » (choix « A ») : un lieu **devient** ce qui
  s'y produit. « Mes PV sont remontés au max ici » → c'est un lieu de soin. On part
  du soin comme premier cas concret, avec l'intention de généraliser au 2e (l'herbe).
- Le squelette de reconnaissance existe déjà mais n'est **nourri par rien** :
  `MapMemory.observe(snapshot, WorldEvent(healed=...))` ajoute déjà le label
  `"healing_spot"`. Ce bout le branche enfin.
- Offsets PV **confirmés** par une source externe (`heatz123/pokeagent-solution`,
  `memory_reader.py`) : dans la struct d'un Pokémon de l'équipe,
  `current_hp` à l'offset `0x56` (86), `max_hp` à `0x58` (88), tous deux `u16`.
  `PARTY_ADDR=0x020244EC`, taille d'un mon = 100, compte à `0x020244E9`. Ces champs
  sont **non chiffrés** (contrairement au bloc chiffré à `0x20`), donc lisibles
  directement. Adresses = faits (aussi dans le décompilateur pokeemerald).

## Non-goals (assumés pour ce bout)

- **Chercher activement un centre inconnu.** Si aucun lieu de soin n'est connu,
  `execute_order(heal)` renvoie `"no_healing_spot_known"`. La recherche active
  (errer pour trouver un centre) = bout suivant. Conséquence : ce bout apprend les
  lieux de soin **déjà utilisés au moins une fois** (cohérent avec « Façon A : il
  faut l'avoir vécu »).
- **Généraliser** (herbe → grind, arène → badge). On garde l'intention en tête
  (structure, nommage sobre) mais on ne code que le soin. La forme générale émergera
  au 2e cas.
- **Brancher le Stratège.** On appelle `execute_order(Order(..., "heal", ...))` à la
  main dans les tests ; le Stratège n'émet pas encore de vrais ordres.
- **Re-cartographier les intérieurs de bâtiments.** On réutilise les portails déjà
  enregistrés ; on n'ajoute pas de survey d'intérieur.

## Vue d'ensemble du flux

```
Stratège : Order(destination=<ignoré pour heal>, mode="heal", combat=...)
     │
     ▼
execute_order (env/orders.py)
  1. spot = un lieu de soin connu (MapMemory.healing_spots())
       aucun → return "no_healing_spot_known"
  2. outcome = travel_to(...jusqu'à (spot.map, spot.cell)...)   # navigation réutilisée
       outcome != "arrived" → return outcome                     # pass-through
  3. return _heal_here(emulator, reader)   # interaction : presser A jusqu'à PV pleins
```

Et **en tâche de fond, pendant que l'Explorateur se déplace normalement** (dans les
boucles `navigate_to` / `map_map`), un `HealWatcher` surveille les PV. Au pas où
l'équipe passe de « pas pleine » à « pleine », on appelle
`memory.observe(snapshot, WorldEvent(healed=True))` → le lieu courant est étiqueté
`"healing_spot"` et sa case mémorisée. C'est **l'apprentissage** du lieu de soin.

## Composants (fichiers)

### 1. `env/game_state.py` — AJOUT : lire les PV de l'équipe

Nouvelle méthode sur `EmeraldReader`, calquée sur `party_levels()` existante :

```python
PARTY_HP_OFFSET = 86      # 0x56, u16 current HP  (confirmé)
PARTY_MAX_HP_OFFSET = 88  # 0x58, u16 max HP      (confirmé)

def party_hp(self) -> list[tuple[int, int]]:
    """Return (current_hp, max_hp) per party member, in order."""
    count = self._read_u8(PARTY_COUNT_ADDR)
    out: list[tuple[int, int]] = []
    for i in range(count):
        base = PARTY_ADDR + i * PARTY_MON_SIZE
        cur = self._read_u16(base + PARTY_HP_OFFSET)
        mx = self._read_u16(base + PARTY_MAX_HP_OFFSET)
        out.append((cur, mx))
    return out
```

(Le nom exact des helpers `_read_u8`/`_read_u16` et des constantes existantes est à
reprendre du fichier réel ; `party_levels()` sert de modèle.)

### 2. `env/heal_detector.py` — NOUVEAU : le guetteur (pur, sans ROM)

```python
def party_is_full(hp: list[tuple[int, int]]) -> bool:
    """True if every living member is at max HP (and the party is non-empty)."""
    if not hp:
        return False
    return all(cur >= mx for cur, mx in hp)


class HealWatcher:
    """Detects the step where the party transitions from not-full to full HP."""

    def __init__(self) -> None:
        self._was_full = True   # start optimistic: no spurious heal on first read

    def observe(self, hp: list[tuple[int, int]]) -> bool:
        """Feed the current party HP. Returns True on the transition to full."""
        full = party_is_full(hp)
        healed = full and not self._was_full
        self._was_full = full
        return healed
```

- **Pourquoi `_was_full = True` au départ** : sinon la toute première lecture (déjà
  pleine) déclencherait un faux « soin ». On ne signale un soin que sur une vraie
  transition « pas plein → plein ».
- Pur, testable sans émulateur.

### 3. `env/map_memory.py` — mémoriser la CASE de soin

Le label `"healing_spot"` existe déjà via `observe(...)`. On ajoute la case où le
soin a eu lieu, pour savoir où retourner.

- Stockage : `MapMemory._healing_cells: dict[map_id, cell]` (privé).
- Dans `observe(snapshot, event)` : si `event.healed`, en plus du label existant,
  faire `self._healing_cells[snapshot.map_id] = snapshot.pos` (last-write-wins).
- Nouvel accesseur :

```python
def healing_spots(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Known healing locations as (map_id, cell), most-recently-learned order not
    guaranteed. Empty if none learned yet."""
    return [(map_id, cell) for map_id, cell in self._healing_cells.items()]
```

(Stockage heal-spécifique volontaire pour ce bout ; on généralisera au 2e cas.)

### 4. `env/orders.py` — remplir le mode `"heal"`

`execute_order` : quand `order.mode == "heal"`, au lieu de `"not_implemented"` :

```python
if order.mode == "heal":
    return _execute_heal(emulator, reader, memory, wallmap, max_hops=max_hops)
```

Nouvelle fonction privée :

```python
def _execute_heal(emulator, reader, memory, wallmap, max_hops=20) -> str:
    spots = memory.healing_spots()
    if not spots:
        return "no_healing_spot_known"
    goal_map, goal_cell = spots[0]   # v1: le premier connu (choix du plus proche = plus tard)
    outcome = travel_to(emulator, reader, memory, wallmap, goal_map, goal_cell,
                        max_hops=max_hops)
    if outcome != "arrived":
        return outcome               # pass-through: unknown_route / unreachable / lost / timeout
    return _heal_here(emulator, reader)
```

Interaction de soin (le seul morceau vraiment nouveau, à valider sur ROM) :

```python
HEAL_PRESS_A_FRAMES = 6
HEAL_RELEASE_FRAMES = 10
HEAL_MAX_PRESSES = 60   # borne (règle code-safety #2)

def _heal_here(emulator, reader) -> str:
    """Press A repeatedly (advancing the nurse dialog) until the party is full."""
    for _ in range(HEAL_MAX_PRESSES):
        if party_is_full(reader.party_hp()):
            return "healed"
        emulator.step(KEY_A, HEAL_PRESS_A_FRAMES)
        emulator.step(0, HEAL_RELEASE_FRAMES)   # release entre appuis (piège debounce GBA)
    return "heal_failed" if not party_is_full(reader.party_hp()) else "healed"
```

- La condition de succès (`party_is_full`) est **le même signal** que le guetteur.
- Le release entre chaque appui A évite la fusion de deux appuis par la GBA (piège
  déjà rencontré sur le Fighter et le navigateur).
- **Risque connu / à tester en premier** : l'orientation du perso. La case mémorisée
  est celle d'où un soin a déjà été déclenché ; en y revenant on devrait faire face
  au bon interlocuteur. Si ça échoue sur ROM, on itérera (mémoriser aussi la
  direction, ou ré-approcher depuis la même case adjacente).

### 5. Câbler le guetteur dans les boucles de déplacement

Pour que les lieux de soin s'apprennent **pendant l'exploration normale**, brancher
un `HealWatcher` dans les boucles live qui steppent l'émulateur :

- `env/live_navigator.py` (`navigate_to`) et/ou `env/map_explorer.py` (`map_map`) :
  à chaque tour de boucle, après le `snapshot`, si `memory is not None`, faire
  `if watcher.observe(reader.party_hp()): memory.observe(snapshot, WorldEvent(healed=True))`.
- Contrainte : **ne rien casser** si `memory is None` (comportement identique à
  aujourd'hui). Le `HealWatcher` est instancié en début de boucle.
- Détail : `navigate_to` reçoit déjà `memory` (param optionnel, P3 étape 2). On
  réutilise ce chemin ; pas de nouvelle signature publique si possible.

> Choix de portée : si le câblage dans les deux boucles alourdit trop, on peut le
> limiter à `map_map` (le mode où l'Explorateur erre vraiment) pour ce bout, et
> l'étendre plus tard. À trancher dans le plan selon la simplicité réelle du code.

## Contrat des sorties de `execute_order(mode="heal")`

- `"no_healing_spot_known"` — aucun lieu de soin appris (territoire inconnu, on
  n'explore pas dans ce bout).
- `"unknown_route"` / `"unreachable"` / `"lost"` / `"timeout"` — pass-through de
  `travel_to` : une jambe de navigation a échoué.
- `"healed"` — arrivé sur place ET PV remontés au max.
- `"heal_failed"` — arrivé sur place mais les PV ne sont pas montés dans le budget
  (interaction ratée : mauvaise orientation, pas la bonne case, etc.).

## Tests

Tous sans ROM sauf le smoke, via des fakes déjà établis dans le projet
(`FakeEmulator` / mondes-fakes jouant émulateur + reader).

1. `tests/test_game_state.py` (ou fichier PV dédié) — `party_hp()` lit
   (current, max) par membre depuis un `FakeEmulator` servant les octets 0x56/0x58.
2. `tests/test_heal_detector.py` :
   - `party_is_full` : équipe pleine → True ; un membre blessé → False ; vide → False.
   - `HealWatcher` : séquence plein→plein = jamais de soin ; plein→blessé→plein =
     un seul `True` au retour au plein ; pas de faux positif au 1er appel.
3. `tests/test_map_memory.py` (+) — `observe(healed=True)` ajoute le label ET
   mémorise la case ; `healing_spots()` la renvoie ; last-write-wins sur re-soin
   même carte.
4. `tests/test_orders.py` (+) :
   - `mode="heal"` sans lieu connu → `"no_healing_spot_known"`.
   - `mode="heal"` avec un lieu connu sur la carte courante → le fake « soigne »
     (PV montent après quelques appuis A) → `"healed"`.
   - `mode="heal"` avec lieu connu mais l'interaction ne soigne jamais →
     `"heal_failed"`.
   - `mode="heal"` avec lieu connu injoignable (fake) → pass-through
     (`"unreachable"`/`"timeout"`).
5. `tests/test_live_navigator.py` ou `test_map_explorer.py` (+) — la boucle apprend
   un lieu de soin quand le fake fait remonter les PV pendant le déplacement ;
   régression : `memory=None` → comportement inchangé.
6. Smoke ROM (différé si pas de savestate adapté) : un état « équipe blessée +
   lieu de soin connu » n'existe peut-être pas encore ; à capturer plus tard. Ne
   bloque pas ce bout.

## Découpage indicatif (pour le plan, TDD)

1. `party_hp()` (+ test).
2. `heal_detector.py` : `party_is_full` + `HealWatcher` (+ tests).
3. `MapMemory` : mémoriser la case + `healing_spots()` (+ tests).
4. `orders.py` : `_execute_heal` + `_heal_here` (+ tests, fakes).
5. Câbler le `HealWatcher` dans la/les boucle(s) de déplacement (+ test + régression).
6. (Optionnel) smoke ROM si un savestate adapté est capturable.
