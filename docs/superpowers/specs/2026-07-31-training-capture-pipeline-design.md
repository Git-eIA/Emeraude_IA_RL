# Design — Training Capture Pipeline (sous-projet 1 : Capture)

Date : 2026-07-31
Statut : validé (brainstorming), prêt pour writing-plans

## Contexte & objectif

On veut produire une vidéo YouTube sur l'apprentissage de l'IA (style « Training AI to
Play Pokémon » sur la version Rouge). La vidéo a besoin de 6 visuels signature :

- **A** — grille d'agents parallèles qui jouent en même temps
- **B** — heatmap d'exploration qui se colore sur la carte au fil du temps
- **C** — carte « fog of war » qui se révèle case par case
- **D** — courbes de progression (reward moyen, jalons, etc.)
- **E** — timeline des jalons « first-time » avec clip du moment
- **F** — avant / après (même scène jouée par un checkpoint tôt vs tard)

**Décision produit** : on ne rejoue PAS le passé. Les entraînements passés n'ont gardé
aucun log de positions/frames. On **prépare l'instrumentation maintenant** pour que le
**prochain** entraînement (et les suivants) enregistre tout proprement. Objectif : au
prochain `train.py`, la data video-ready sort sur disque toute seule.

**Découpage en 2 sous-projets** :

1. **Capture** (CE document, urgent — doit exister AVANT le prochain train) : instrumenter
   l'entraînement pour émettre une data structurée sur disque.
2. **Rendu** (spec séparée, plus tard) : outils offline qui transforment la data capturée
   en les 6 visuels. Ne bloque aucun train ; a besoin de vraies captures pour être testé.

Cette spec fige un **contrat de données** clair pour que le sous-projet Rendu soit
buildable ensuite sans re-toucher l'entraînement.

## Contraintes

- **Volume** : numérique (positions, reward, jalons) minuscule → loggé à chaque pas de
  chaque env. Images lourdes → **échantillonnées** + **clips** sur événements. Budget
  disque confortable : dizaines de Go.
- **Non-régression RL** : la capture ne doit JAMAIS faire planter ou ralentir
  significativement l'entraînement, ni changer la sémantique RL (reward, observation,
  action). Tout l'I/O de capture est fail-safe.
- **Cible** : le trainer Explorer (`agent/train.py`, PPO `CnnPolicy`, `SubprocVecEnv`).
  Le même `RecorderCallback` doit rester réutilisable pour les autres trainers plus tard,
  mais on ne câble que `train.py` en v1.
- **Convention de test du projet** : unitaires sans ROM via fakes ; smoke ROM gated par
  `POKEMON_EMERALD_ROM`.

## Approche retenue

**Approche 1 — un seul `RecorderCallback` SB3** dans le processus principal. Il lit les
`infos` / `rewards` des envs (déjà transférés par SB3 via la frontière de processus,
gratuit) pour le numérique, et appelle `env_method("render")` pour les frames. Aligné sur
le pas global PPO → courbes propres. Clips « post-roll » (à partir du moment du jalon).

Rejetées : Approche 2 (wrapper par env — ne connaît pas le pas global, coordination
pénible) et Approche 3 (hybride — pièces mobiles en trop pour le seul gain du pré-roll).

## Architecture — 3 composants

### 1. Enrichir `info` (`env/pokemon_env.py`)

`PokemonEmeraldEnv._info()` expose aujourd'hui `visited_tiles`, `badges`, `map`,
`milestones`. Ajouter deux champs, lisibles depuis `PlayerState` (qui a déjà `x`, `y`) :

- `pos: tuple[int, int] | None` = `(state.x, state.y)` (None si `state is None`)
- `step: int` = `self._step_count`

Seule touche à l'env. **Zéro** changement à la logique RL / reward / observation.

### 2. `RecorderCallback` (nouveau — `env/capture/recorder.py`)

Un `stable_baselines3.common.callbacks.BaseCallback`.

**Paramètres (défauts raisonnables)** :

- `run_dir: Path` — racine du run (`captures/<run_id>/`)
- `capture_every: int = 200` — période d'échantillonnage des frames (en pas globaux PPO)
- `clip_len: int = 48` — nb de pas post-roll enregistrés en clip après un jalon
  (≈ quelques secondes de jeu)
- `max_frame_gb: float = 20.0` — cap disque souple pour les images
- `frame_format: str = "jpg"` — encodage des frames (JPEG pour la taille)

**Cycle de vie** :

- `_on_training_start` :
  - crée l'arbo `run_dir/` (`frames/`, `clips/`, `checkpoints/`)
  - ouvre `steps.csv` et `milestones.csv` en append avec en-tête
  - écrit `run.json` : `run_id`, `git_commit`, `argv`, `n_envs`, `total_timesteps`,
    `start_wall_time`, `rom`, `initial_state`, `schema_version` (= `1`)
  - initialise l'état interne : `_seen_milestones: list[set[str]]` (un set/env),
    `_clip_remaining: list[int]` (un compteur/env), `_frame_bytes: int`,
    `_disabled_frames: bool = False`

- `_on_step` (appelé une fois par pas de vec-env ; `len(infos) == n_envs`) :
  1. lit `self.locals["infos"]` et `self.locals["rewards"]`
  2. **numérique** : pour chaque env `i`, écrit une ligne dans `steps.csv` :
     `t=self.num_timesteps, env=i, map_g, map_n, x, y, reward, visited_tiles`
     (map/pos depuis `info["map"]` / `info["pos"]` ; ligne écrite même si `pos is None`,
     champs vides)
  3. **jalons** : pour chaque env, `new = set(info["milestones"]) - _seen_milestones[i]` ;
     pour chaque jalon nouveau → ligne dans `milestones.csv`
     (`t, env, milestone, wall_time`) et arme un clip : `_clip_remaining[i] = clip_len`,
     mémorise le `t0`/nom pour le dossier ; met à jour `_seen_milestones[i]`
  4. **frames échantillonnées** : si `self.num_timesteps` franchit un multiple de
     `capture_every` (et frames non désactivées) → `self.training_env.env_method("render")`
     pour tous les envs → un JPEG par env dans `frames/env{i}/{t:09d}.jpg`
  5. **clips** : pour chaque env avec `_clip_remaining[i] > 0` → `env_method("render",
     indices=[i])` → JPEG dans `clips/{milestone}_{t0}/env{i}/{seq:04d}.jpg`, décrémente
  6. **cap disque** : accumule les octets écrits ; si `_frame_bytes > max_frame_gb*1e9` →
     `_disabled_frames = True` (on garde le numérique, on arrête frames+clips) + warning
  7. retourne toujours `True`

- `_on_training_end` : flush/close des CSV, finalise `run.json`
  (`end_wall_time`, `final_timestep`)

**Fail-safe (règle non-négociable)** : chaque bloc d'I/O de capture (`_write_step`,
`_write_milestone`, `_grab_frames`, `_grab_clip`) est enveloppé dans un try/except qui
loggue un warning via `logging.getLogger("env.capture")` et incrémente un compteur
d'erreurs. Au-delà d'un seuil (`_MAX_ERRORS = 20`), la capture se désactive entièrement
(`_disabled = True`) et l'entraînement continue intact. Un render qui lève ne saute que la
frame concernée.

**Coût maîtrisé** : `env_method("render")` (aller-retour inter-processus) n'est appelé
qu'à la cadence d'échantillonnage OU pendant une fenêtre de clip (rare) — jamais à chaque
pas pour tous les envs.

### 3. Câblage `agent/train.py`

- Nouveaux flags argparse : `--run-id` (défaut = horodatage `YYYYmmdd-HHMMSS`),
  `--capture` / `--no-capture` (défaut : capture ON), `--capture-every` (défaut 200),
  `--clip-len` (défaut 48), `--max-frame-gb` (défaut 20.0)
- `run_dir = Path("captures") / run_id`
- Si capture ON : construire `RecorderCallback(run_dir=run_dir, ...)` et
  `CallbackList([checkpoints, recorder])` ; sinon garder `checkpoints` seul
- `CheckpointCallback.save_path = run_dir / "checkpoints"` (chaque run auto-contenu → l'avant/après F trouve ses checkpoints dans le dossier du run). `ppo_emerald_final` sauvé dans `run_dir/checkpoints/` aussi
- `tensorboard_log="runs"` conservé, mais passer `tb_log_name=run_id` à `model.learn`
  (corrige l'empilement de tout dans `runs/PPO_0`)

## Contrat de données — `captures/<run_id>/`

Interface figée pour le sous-projet Rendu. `schema_version` dans `run.json` gouverne toute
évolution future.

```
run.json                          # métadonnées du run (voir champs ci-dessus)
steps.csv                         # header: t,env,map_g,map_n,x,y,reward,visited_tiles
milestones.csv                    # header: t,env,milestone,wall_time
frames/envK/<t>.jpg               # frames échantillonnées par env  (→ grille A, réf F)
clips/<milestone>_<t0>/envK/<seq>.jpg   # clips post-roll par événement  (→ timeline E)
checkpoints/ppo_emerald_<n>_steps.zip   # checkpoints périodiques  (→ avant/après F)
checkpoints/ppo_emerald_final.zip
```

**Comment les 6 visuels se dérivent de ce contrat** (côté Rendu, plus tard) :

- **A** (grille) ← `frames/env*/` alignées par `t`
- **B** (heatmap) ← `steps.csv` (x,y,map) agrégés dans le temps
- **C** (fog of war) ← `steps.csv` : cases visitées révélées progressivement (approximation
  gratuite ; le vrai `WallMap` fog viendrait d'un run `survey_world` instrumenté — hors v1)
- **D** (courbes) ← `steps.csv` (reward, visited_tiles) + `milestones.csv` (+ TensorBoard
  propre par run)
- **E** (timeline) ← `milestones.csv` + `clips/`
- **F** (avant/après) ← `checkpoints/` rejoués sur un savestate (outil de rendu dédié)

## Gestion d'erreur (récap)

- I/O capture fail-safe → warning + désactivation après seuil ; l'entraînement continue.
- Render d'un env qui lève → frame sautée, pas de crash.
- Cap disque souple → arrêt des images, numérique conservé.
- `pos is None` (relocalisation SaveBlock) → ligne écrite avec x,y vides.

## Tests

Convention projet : unitaires sans ROM via fakes ; smoke ROM gated.

- `tests/test_recorder.py` (sans ROM) via un faux `training_env` exposant `env_method`
  (render → array factice) + faux `self.locals` (`infos`, `rewards`) et `num_timesteps` :
  - `steps.csv` reçoit une ligne par env par pas, bon schéma
  - un nouveau jalon dans `info["milestones"]` → ligne dans `milestones.csv` + dossier de
    clip créé ; un jalon déjà vu → pas de doublon
  - frames écrites uniquement aux multiples de `capture_every`, une par env
  - fenêtre de clip : `clip_len` frames écrites par env armé, puis arrêt
  - fail-safe : un render qui lève ne fait pas planter `_on_step` (retourne `True`)
  - cap disque : au-delà de `max_frame_gb`, plus de frames, numérique continue
  - `run.json` contient les champs attendus + `schema_version`
- `tests/test_pokemon_env.py` (existant, sans ROM) : `info` contient `pos` et `step`
- `tests/test_recorder_rom.py` (gated `POKEMON_EMERALD_ROM`) : mini-train réel
  (`--timesteps 512 --envs 1 --capture`) → `captures/<run_id>/` existe, `steps.csv` non
  vide, `run.json` valide, au moins une frame écrite

## Non-goals (v1)

- Les outils de rendu (sous-projet 2 — spec séparée)
- Vrai fog-of-war `WallMap` (on prend les cases visitées de `steps.csv`)
- Clips « pré-roll » (on fait post-roll)
- Instrumenter les trainers Fighter/Strategist (même callback réutilisable plus tard)
- Encodage mp4 (on garde des JPEG ; l'encodage est une préoccupation de la phase Rendu)
- Format columnar/parquet pour `steps.csv` (CSV suffit ; ~centaines de Mo sur un gros run)
```
