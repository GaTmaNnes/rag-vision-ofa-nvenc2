//! Scan de corpus — niveau N0 (EXISTS) + classification.
//!
//! Contrairement a la version actuelle de jarvis-rs (corpus_manifest.rs), ce module
//! NE PAS ignorer les videos/images/PDF : on veut au minimum savoir qu'ils existent
//! (c'est le principe N0 EXISTS de PRINCIPE.md). Le contenu binaire brut (exe, dll,
//! archives) reste ignore, lui, parce qu'il n'a rien a apporter a un pipeline RAG.

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::path::Path;
use std::time::Instant;
use walkdir::WalkDir;

/// Au-dela de cette taille, on ne hashe que le debut/fin du fichier (fingerprint
/// approximatif) plutot que tout le contenu, pour rester rapide sur un gros corpus.
const FULL_HASH_LIMIT: u64 = 20 * 1024 * 1024; // 20 Mo
const CHUNK_SIZE: usize = 64 * 1024; // 64 Ko

/// Hash de contenu reel (contrairement au `size:size` place-holder trouve dans la
/// version actuelle de jarvis-rs, qui ne distingue jamais deux fichiers de meme
/// taille). Pas cryptographique — juste assez robuste pour de la dedup pratique.
fn content_hash(path: &Path, size: u64) -> Option<u64> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = DefaultHasher::new();
    size.hash(&mut hasher);

    if size <= FULL_HASH_LIMIT {
        let mut buf = Vec::with_capacity(size as usize);
        file.read_to_end(&mut buf).ok()?;
        buf.hash(&mut hasher);
    } else {
        // Fingerprint : premier + dernier chunk seulement.
        let mut head = vec![0u8; CHUNK_SIZE];
        let n = file.read(&mut head).ok()?;
        head[..n].hash(&mut hasher);

        use std::io::{Seek, SeekFrom};
        if size > CHUNK_SIZE as u64 {
            let tail_start = size.saturating_sub(CHUNK_SIZE as u64);
            file.seek(SeekFrom::Start(tail_start)).ok()?;
            let mut tail = vec![0u8; CHUNK_SIZE];
            let n = file.read(&mut tail).ok()?;
            tail[..n].hash(&mut hasher);
        }
    }
    Some(hasher.finish())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Text,
    Code,
    Video,
    Image,
    Pdf,
    Audio,
    Other,
}

impl Kind {
    fn from_ext(ext: &str) -> Kind {
        match ext.to_lowercase().as_str() {
            "txt" | "md" | "rst" | "json" | "yaml" | "yml" | "toml" | "csv" => Kind::Text,
            "rs" | "py" | "c" | "h" | "hpp" | "cpp" | "cc" | "cu" | "cuh" | "js" | "ts"
            | "java" | "go" | "rb" | "php" | "sh" | "html" | "css" => Kind::Code,
            "mp4" | "avi" | "mkv" | "mov" | "webm" | "wmv" => Kind::Video,
            "png" | "jpg" | "jpeg" | "gif" | "bmp" | "webp" | "svg" => Kind::Image,
            "pdf" => Kind::Pdf,
            "wav" | "mp3" | "flac" | "aac" | "ogg" => Kind::Audio,
            _ => Kind::Other,
        }
    }

    /// Est-ce que ce type de fichier peut alimenter l'index BM25 directement ?
    pub fn is_indexable_as_text(&self) -> bool {
        matches!(self, Kind::Text | Kind::Code)
    }
}

/// Extensions binaires qu'on ignore vraiment (aucune valeur informationnelle pour JARVIX).
const IGNORE_EXT: &[&str] = &[
    "exe", "dll", "so", "dylib", "obj", "o", "lib", "a", "pdb", "zip", "7z", "rar", "tar",
    "gz", "bz2", "xz", "pyc", "pyo", "bin", "dat", "iso",
];

const IGNORE_DIRS: &[&str] = &[
    ".git", "node_modules", "target", "__pycache__", ".venv", "venv", ".cache",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRecord {
    pub path: String,
    pub size: u64,
    pub kind: Kind,
    /// Hash de contenu (voir `content_hash`). `None` si le fichier n'a pas pu etre lu.
    pub hash: Option<u64>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ScanStats {
    pub n_files: usize,
    pub n_ignored: usize,
    pub n_dedup: usize,
    pub total_size_mb: f64,
    pub scan_time_s: f64,
    pub by_kind: HashMap<String, usize>,
}

pub struct Corpus {
    pub files: Vec<FileRecord>,
}

impl Corpus {
    pub fn scan(root: &str) -> anyhow::Result<(Corpus, ScanStats)> {
        let t0 = Instant::now();
        if !Path::new(root).exists() {
            anyhow::bail!("{root} n'existe pas");
        }

        let mut files = Vec::new();
        let mut n_ignored = 0usize;
        let mut n_dedup = 0usize;
        let mut by_kind: HashMap<String, usize> = HashMap::new();
        let mut total_size: u64 = 0;
        let mut seen_hashes: HashSet<u64> = HashSet::new();

        let walker = WalkDir::new(root).into_iter().filter_entry(|e| {
            if e.file_type().is_dir() {
                let name = e.file_name().to_string_lossy();
                return !IGNORE_DIRS.iter().any(|d| name == *d);
            }
            true
        });

        for entry in walker.filter_map(|e| e.ok()) {
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();

            if IGNORE_EXT.contains(&ext.as_str()) {
                n_ignored += 1;
                continue;
            }

            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            let kind = Kind::from_ext(&ext);
            let hash = content_hash(path, size);

            // Dedup reelle : deux fichiers avec le meme hash de contenu ne sont
            // comptes qu'une fois (contrairement au hash `size:size` de jarvis-rs,
            // qui confondait tous les fichiers de meme taille).
            if let Some(h) = hash {
                if !seen_hashes.insert(h) {
                    n_dedup += 1;
                    continue;
                }
            }

            total_size += size;
            *by_kind.entry(format!("{kind:?}")).or_insert(0) += 1;

            files.push(FileRecord {
                path: path.to_string_lossy().to_string(),
                size,
                kind,
                hash,
            });
        }

        let stats = ScanStats {
            n_files: files.len(),
            n_ignored,
            n_dedup,
            total_size_mb: total_size as f64 / (1024.0 * 1024.0),
            scan_time_s: t0.elapsed().as_secs_f64(),
            by_kind,
        };

        Ok((Corpus { files }, stats))
    }

    pub fn text_files(&self) -> Vec<&FileRecord> {
        self.files
            .iter()
            .filter(|f| f.kind.is_indexable_as_text())
            .collect()
    }

    pub fn video_files(&self) -> Vec<&FileRecord> {
        self.files.iter().filter(|f| f.kind == Kind::Video).collect()
    }

    /// Sauvegarde le manifest (liste des FileRecord) en JSON.
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let json = serde_json::to_string(&self.files)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Recharge un manifest sauvegarde par `save`. Ne re-scanne PAS le disque —
    /// c'est un instantane du dernier scan, potentiellement perime si les fichiers
    /// ont change depuis.
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let files: Vec<FileRecord> = serde_json::from_str(&json)?;
        Ok(Corpus { files })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn classifies_extensions() {
        assert_eq!(Kind::from_ext("md"), Kind::Text);
        assert_eq!(Kind::from_ext("rs"), Kind::Code);
        assert_eq!(Kind::from_ext("mp4"), Kind::Video);
        assert_eq!(Kind::from_ext("png"), Kind::Image);
        assert_eq!(Kind::from_ext("pdf"), Kind::Pdf);
        assert_eq!(Kind::from_ext("wav"), Kind::Audio);
        assert_eq!(Kind::from_ext("xyz_inconnu"), Kind::Other);
    }

    #[test]
    fn only_text_and_code_are_indexable() {
        assert!(Kind::Text.is_indexable_as_text());
        assert!(Kind::Code.is_indexable_as_text());
        assert!(!Kind::Video.is_indexable_as_text());
        assert!(!Kind::Image.is_indexable_as_text());
    }

    #[test]
    fn dedup_detects_identical_content_regardless_of_name() {
        let dir = std::env::temp_dir().join(format!("jarvis_mini_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Deux fichiers de MEME taille et MEME contenu, noms differents.
        let mut f1 = std::fs::File::create(dir.join("a.txt")).unwrap();
        f1.write_all(b"contenu identique pour test dedup").unwrap();
        let mut f2 = std::fs::File::create(dir.join("b.txt")).unwrap();
        f2.write_all(b"contenu identique pour test dedup").unwrap();
        // Un troisieme fichier de la MEME taille mais contenu DIFFERENT :
        // le bug `size:size` de jarvis-rs les aurait confondus avec les deux
        // premiers puisqu'ils ont exactement la meme longueur.
        let mut f3 = std::fs::File::create(dir.join("c.txt")).unwrap();
        f3.write_all(b"contenu different pour test dedupp").unwrap();

        let (_corpus, stats) = Corpus::scan(dir.to_str().unwrap()).unwrap();

        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(stats.n_files, 2, "un doublon exact doit etre dedup");
        assert_eq!(stats.n_dedup, 1);
    }
}
