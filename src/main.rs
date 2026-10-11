mod bm25;
mod corpus;
mod video;

use clap::Parser;
use corpus::Corpus;
use std::fs;

/// JARVIX mini — demo compacte : scan + BM25 + ingestion video honnete.
#[derive(Parser)]
#[command(name = "jarvis-mini")]
#[command(about = "Coeur RAG minimal + ingestion video (voir README.md)")]
struct Cli {
    /// Scanne un dossier et affiche l'inventaire (N0), video/image/pdf inclus.
    #[arg(long)]
    scan: Option<String>,

    /// Scanne + indexe les fichiers texte/code en BM25, puis interroge.
    #[arg(long)]
    ingest: Option<String>,

    /// Requete a executer. Avec --ingest : indexe puis interroge. Seule (sans
    /// --ingest) et avec --index-path : recharge un index deja sauvegarde,
    /// sans rescanner le disque.
    #[arg(long)]
    query: Option<String>,

    /// Dossier ou sauvegarder/charger l'index (bm25.json + manifest.json).
    /// Avec --ingest : sauvegarde apres indexation. Avec --query seul : charge
    /// depuis ce dossier au lieu de rescanner.
    #[arg(long)]
    index_path: Option<String>,

    /// Affiche les metadonnees d'une video (largeur/hauteur/fps/duree/codec).
    #[arg(long)]
    probe_video: Option<String>,

    /// Extrait des frames d'une video et les ecrit en .ppm dans --out-dir.
    #[arg(long)]
    extract_frames: Option<String>,

    #[arg(long, default_value_t = 0)]
    start_frame: usize,

    #[arg(long, default_value_t = 5)]
    count: usize,

    #[arg(long, default_value = "frames_out")]
    out_dir: String,

    /// Tente le decodage materiel (NVDEC) si ffmpeg le supporte. Voir video.rs
    /// pour la note d'honnetete sur CPU vs GPU.
    #[arg(long, default_value_t = false)]
    hwaccel: bool,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if let Some(path) = cli.scan {
        return cmd_scan(&path);
    }

    if let Some(path) = cli.ingest {
        return cmd_ingest(&path, cli.query.as_deref(), cli.index_path.as_deref());
    }

    if let Some(q) = cli.query {
        // --query seul (sans --ingest) : necessite un index deja sauvegarde.
        let index_path = cli.index_path.ok_or_else(|| {
            anyhow::anyhow!("--query seul necessite --index-path <dossier> (index deja sauvegarde via --ingest)")
        })?;
        return cmd_query_only(&index_path, &q);
    }

    if let Some(path) = cli.probe_video {
        return cmd_probe_video(&path);
    }

    if let Some(path) = cli.extract_frames {
        return cmd_extract_frames(&path, cli.start_frame, cli.count, &cli.out_dir, cli.hwaccel);
    }

    println!("jarvis-mini : rien a faire. Essaie --scan <dossier>, --ingest <dossier> --query <texte>,");
    println!("              --probe-video <fichier.mp4>, ou --extract-frames <fichier.mp4>.");
    Ok(())
}

fn cmd_scan(path: &str) -> anyhow::Result<()> {
    let (corpus, stats) = Corpus::scan(path)?;
    println!("=== SCAN ===");
    println!("Chemin       : {path}");
    println!(
        "Fichiers     : {} ({} ignores, {} dedupliques)",
        stats.n_files, stats.n_ignored, stats.n_dedup
    );
    println!("Taille       : {:.1} Mo", stats.total_size_mb);
    println!("Temps        : {:.3}s", stats.scan_time_s);
    println!("Par type     :");
    let mut kinds: Vec<_> = stats.by_kind.iter().collect();
    kinds.sort_by(|a, b| b.1.cmp(a.1));
    for (kind, count) in kinds {
        println!("  {kind:<8} : {count}");
    }
    let n_video = corpus.video_files().len();
    if n_video > 0 {
        println!();
        println!("{n_video} fichier(s) video detecte(s) — utilise --probe-video / --extract-frames pour aller plus loin.");
    }
    Ok(())
}

fn cmd_ingest(path: &str, query: Option<&str>, index_path: Option<&str>) -> anyhow::Result<()> {
    let (corpus, stats) = Corpus::scan(path)?;
    let text_files = corpus.text_files();

    let mut index = bm25::Bm25Index::new();
    let mut n_read = 0usize;
    for rec in &text_files {
        if rec.size > 500_000 {
            continue; // fichier texte trop gros, on l'ignore pour ce mini-index
        }
        if let Ok(content) = fs::read_to_string(&rec.path) {
            if content.len() > 20 {
                index.add(&content, &rec.path);
                n_read += 1;
            }
        }
    }

    println!("=== INGEST ===");
    println!(
        "Fichiers scannes : {} ({} ignores, {} dedupliques)",
        stats.n_files, stats.n_ignored, stats.n_dedup
    );
    println!("Fichiers texte   : {}", text_files.len());
    println!("Documents BM25   : {} (dont {n_read} lus)", index.len());

    if let Some(dir) = index_path {
        fs::create_dir_all(dir)?;
        index.save(&format!("{dir}/bm25.json"))?;
        corpus.save(&format!("{dir}/manifest.json"))?;
        println!("Index sauvegarde : {dir}/bm25.json + {dir}/manifest.json");
    }

    if let Some(q) = query {
        println!();
        println!("=== QUERY: {q} ===");
        for r in index.query(q, 5) {
            println!("  [{:.4}] {} — {}", r.score, r.source, r.excerpt);
        }
    }
    Ok(())
}

fn cmd_query_only(index_dir: &str, query: &str) -> anyhow::Result<()> {
    let bm25_path = format!("{index_dir}/bm25.json");
    let manifest_path = format!("{index_dir}/manifest.json");

    let t0 = std::time::Instant::now();
    let mut index = bm25::Bm25Index::load(&bm25_path)?;
    let load_time = t0.elapsed().as_secs_f64();

    println!("=== QUERY (index charge, pas de rescan) ===");
    println!("Index      : {bm25_path} ({} docs, charge en {load_time:.4}s)", index.len());

    if let Ok(corpus) = corpus::Corpus::load(&manifest_path) {
        println!(
            "Manifest   : {manifest_path} ({} fichiers connus au dernier scan)",
            corpus.files.len()
        );
    }

    println!("Requete    : {query}");
    for r in index.query(query, 5) {
        println!("  [{:.4}] {} — {}", r.score, r.source, r.excerpt);
    }
    Ok(())
}

fn cmd_probe_video(path: &str) -> anyhow::Result<()> {
    let src = video::VideoSource::open(path)?;
    println!("=== PROBE VIDEO ===");
    println!("Fichier   : {path}");
    println!("Resolution: {}x{}", src.meta.width, src.meta.height);
    println!("FPS       : {:.2}", src.meta.fps);
    println!("Duree     : {:.2}s", src.meta.duration_s);
    println!("Codec     : {}", src.meta.codec);
    let approx_frames = (src.meta.duration_s * src.meta.fps).round() as u64;
    println!("~Frames   : {approx_frames} (estime duree x fps)");
    Ok(())
}

fn cmd_extract_frames(path: &str, start: usize, count: usize, out_dir: &str, hwaccel: bool) -> anyhow::Result<()> {
    let src = video::VideoSource::open(path)?;
    fs::create_dir_all(out_dir)?;

    println!("=== EXTRACT FRAMES ===");
    println!("Fichier    : {path}");
    println!("Frames     : {start}..{}", start + count);
    println!("Hwaccel    : {hwaccel}");

    let t0 = std::time::Instant::now();
    let frames = match src.extract_frames(start, count, hwaccel) {
        Ok(f) => f,
        Err(e) if hwaccel => {
            // Echec possible du chemin NVDEC (pas de GPU, driver absent, etc.) --
            // message actionnable au lieu de juste remonter l'erreur brute.
            eprintln!("Echec avec --hwaccel ({e}).");
            eprintln!("-> Cause probable : pas de GPU NVIDIA / decodeur cuda indisponible sur cette machine.");
            eprintln!("-> Relance sans --hwaccel pour le chemin CPU (garanti).");
            return Err(e);
        }
        Err(e) => return Err(e),
    };
    let elapsed = t0.elapsed().as_secs_f64();

    for f in &frames {
        let out_path = format!("{out_dir}/frame_{:05}.ppm", f.index);
        f.save_ppm(&out_path)?;
    }

    println!("Recues     : {} frame(s)", frames.len());
    println!("Temps      : {elapsed:.3}s ({:.1} fps)", frames.len() as f64 / elapsed.max(0.001));
    println!("Ecrites    : {out_dir}/frame_*.ppm");
    println!();
    println!("Rappel : ces buffers sont en memoire hote (CPU), pas des tenseurs GPU —");
    println!("voir la note en tete de src/video.rs.");
    Ok(())
}
