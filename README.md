# jarvis-mini

Mini version compacte de JARVIX, en Rust, construite et **compilée/testée dans cette
session** (pas seulement écrite). Reprend le cœur validé de `jarvis-rs` (scan + BM25)
et y ajoute une ingestion vidéo honnête, tirée de la leçon de la longue session de
debug torchcodec/ffmpeg qu'on a passée en revue dans la conversation.

Projet **séparé** de `jarvis-rs` — rien n'a été modifié dans le dossier original,
conformément à la consigne.

## Ce qui est dedans

- `src/corpus.rs` — scan de dossier (N0 EXISTS). Contrairement à la version actuelle
  de `jarvis-rs`, les vidéos/images/PDF ne sont **pas** ignorés — on garde au moins
  leur existence, taille et type, comme le prévoit `PRINCIPE.md`. **Dédup réelle** :
  hash de contenu (pas le `size:size` placeholder de `jarvis-rs`), full hash pour les
  fichiers < 20 Mo, fingerprint début/fin au-delà.
- `src/bm25.rs` — même algorithme BM25 Okapi que `jarvis-rs`, nettoyé dans son propre
  module. **Persistance** : `save()`/`load()` sérialisent les documents en JSON ;
  au chargement, l'index (term_freq/doc_freq) est reconstruit à partir du texte brut.
- `src/video.rs` — métadonnées via `ffprobe` (subprocess), extraction de frames via
  **un seul appel `ffmpeg`** pour tout un lot (pas un process par frame — c'était la
  cause du ralentissement ×85 mesuré dans les scripts Python examinés plus tôt).
  Retourne des buffers RGB24 en `Vec<u8>`, en mémoire CPU. Échec `--hwaccel` géré
  proprement (message actionnable, pas de panique).
- `src/main.rs` — CLI : `--scan`, `--ingest [--index-path DIR]`, `--query` (avec
  `--ingest` pour indexer+interroger, ou seul + `--index-path` pour interroger un
  index déjà sauvegardé sans rescanner), `--probe-video`, `--extract-frames`.

## Ce qui est délibérément honnête (contrairement aux docs Python examinées)

Le module vidéo **n'affirme jamais** qu'un buffer est "déjà sur le GPU". C'est
l'erreur trouvée à plusieurs reprises dans `ffmpeg_nvdec_decoder.py`,
`FFmpegCUDADecoder.decode_all()`, et le faux snippet `from nvidia import cuda`
(package qui n'existe pas — le vrai est `PyNvVideoCodec`). Ici : les octets passent
par un pipe stdout, donc mémoire hôte, point final. Un flag `--hwaccel` existe pour
demander le décodage matériel (NVDEC) à ffmpeg, mais le commentaire dans le code est
clair sur le fait que ça n'implique pas que le résultat reste sur GPU.

Le seek (`-ss` avant `-i`) est rapide mais précis seulement à la keyframe la plus
proche — documenté comme tel, pas caché.

## Limites assumées (pour rester "mini")

- Pas de SNN/CUDA/GNW — ce serait un portage Python→Rust bien trop lourd pour une
  mini version ; hors scope ici.
- Extraction vidéo = métadonnées + frames brutes seulement, pas de pipeline vision/VLM
  derrière (rien dans `jarvis-rs` ni dans le mini pour consommer ces frames pour le
  moment).
- Le manifest chargé par `--query --index-path` est un instantané du dernier scan —
  pas revalidé contre le disque (un fichier modifié entre-temps ne sera pas détecté
  tant qu'on ne relance pas `--ingest`).
- `--hwaccel` n'a pas pu être testé sur du vrai matériel NVIDIA (ce sandbox n'a pas de
  GPU) — seul le chemin d'échec propre est validé, pas le chemin de succès NVDEC.

## Build & test (déjà fait dans cette session)

```bash
cargo build --release      # compile sans warning
cargo test                  # 12 tests, tous verts

./target/release/jarvis-mini --scan mon_dossier
./target/release/jarvis-mini --ingest mon_dossier --query "ma requete" --index-path index/
./target/release/jarvis-mini --query "ma requete" --index-path index/   # sans rescanner
./target/release/jarvis-mini --probe-video video.mp4
./target/release/jarvis-mini --extract-frames video.mp4 --start-frame 5 --count 3 --out-dir frames/
```

Testé bout en bout avec une vidéo générée par `ffmpeg -f lavfi testsrc` (320×240,
25fps, 2s) :
- scan détecte le fichier vidéo, dédup vérifiée avec un doublon exact (nom différent,
  contenu identique) → correctement compté une seule fois ;
- `--probe-video` lit correctement résolution/fps/durée/codec ;
- `--extract-frames` produit des `.ppm` valides (vérifiés avec `file` — Netpbm pixmap,
  pixels non uniformes) en un seul appel ffmpeg, y compris le cas limite où on demande
  plus de frames qu'il n'en reste en fin de vidéo ;
- `--ingest --index-path` puis `--query --index-path` (sans rescan) : round-trip
  validé, l'index rechargé retrouve les mêmes résultats ;
- `--hwaccel` sur cette machine (pas de GPU NVIDIA) échoue proprement avec message
  actionnable et code de sortie 1, au lieu de planter — le chemin CPU (sans
  `--hwaccel`) fonctionne sur le même fichier juste après.

Nécessite `ffmpeg`/`ffprobe` dans le PATH.
