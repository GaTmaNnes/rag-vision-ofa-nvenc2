//! Index BM25 Okapi — repris du design deja valide dans jarvis-rs (memory.rs).
//!
//! Persistance : on ne serialise QUE les documents bruts (texte + source), pas les
//! tables term_freq/doc_freq calculees -- elles sont reconstruites au chargement
//! via `build()`. C'est un choix delibere : le cout de reconstruction est faible
//! (voir tests), et ca evite de desynchroniser l'index si le format de tokenisation
//! change entre deux versions du binaire.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Doc {
    pub id: usize,
    pub text: String,
    pub source: String,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub score: f32,
    pub excerpt: String,
    pub source: String,
}

pub struct Bm25Index {
    docs: Vec<Doc>,
    avgdl: f64,
    term_freq: HashMap<String, HashMap<usize, usize>>,
    doc_freq: HashMap<String, usize>,
    k1: f64,
    b: f64,
    dirty: bool,
}

impl Bm25Index {
    pub fn new() -> Self {
        Self {
            docs: Vec::new(),
            avgdl: 0.0,
            term_freq: HashMap::new(),
            doc_freq: HashMap::new(),
            k1: 1.5,
            b: 0.75,
            dirty: false,
        }
    }

    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// API publique pratique (utilisee dans les tests) meme si le CLI ne l'appelle
    /// pas directement aujourd'hui.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Sauvegarde les documents bruts en JSON (pas l'index calcule, voir note en tete
    /// de fichier).
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let json = serde_json::to_string(&self.docs)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Recharge des documents depuis le JSON sauvegarde par `save`, et reconstruit
    /// l'index (term_freq/doc_freq) a partir d'eux.
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let docs: Vec<Doc> = serde_json::from_str(&json)?;

        let mut index = Self::new();
        for doc in docs {
            index.add(&doc.text, &doc.source);
        }
        index.build();
        Ok(index)
    }

    pub fn add(&mut self, text: &str, source: &str) {
        let id = self.docs.len();
        self.docs.push(Doc {
            id,
            text: text.to_string(),
            source: source.to_string(),
        });
        self.dirty = true;
    }

    fn build(&mut self) {
        if !self.dirty {
            return;
        }
        self.term_freq.clear();
        self.doc_freq.clear();
        let mut total_len = 0usize;

        for doc in &self.docs {
            let tokens = tokenize(&doc.text);
            total_len += tokens.len();
            let mut seen = std::collections::HashSet::new();
            for t in &tokens {
                *self
                    .term_freq
                    .entry(t.clone())
                    .or_default()
                    .entry(doc.id)
                    .or_insert(0) += 1;
                if seen.insert(t.clone()) {
                    *self.doc_freq.entry(t.clone()).or_insert(0) += 1;
                }
            }
        }
        self.avgdl = if !self.docs.is_empty() {
            total_len as f64 / self.docs.len() as f64
        } else {
            0.0
        };
        self.dirty = false;
    }

    pub fn query(&mut self, text: &str, k: usize) -> Vec<SearchResult> {
        self.build();
        let n_docs = self.docs.len() as f64;
        let tokens = tokenize(text);
        let mut scores: HashMap<usize, f64> = HashMap::new();

        for token in &tokens {
            let df = *self.doc_freq.get(token).unwrap_or(&0) as f64;
            if df == 0.0 {
                continue;
            }
            let idf = ((n_docs - df + 0.5) / (df + 0.5) + 1.0).ln();

            if let Some(postings) = self.term_freq.get(token) {
                for (&doc_id, &tf) in postings {
                    let doc_len = tokenize(&self.docs[doc_id].text).len() as f64;
                    let tf_norm = tf as f64 * (self.k1 + 1.0)
                        / (tf as f64 + self.k1 * (1.0 - self.b + self.b * doc_len / self.avgdl.max(1.0)));
                    *scores.entry(doc_id).or_insert(0.0) += idf * tf_norm;
                }
            }
        }

        let mut results: Vec<SearchResult> = scores
            .into_iter()
            .map(|(id, score)| {
                let doc = &self.docs[id];
                SearchResult {
                    score: score as f32,
                    excerpt: doc.text.chars().take(160).collect(),
                    source: doc.source.clone(),
                }
            })
            .collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() >= 2 && s.len() <= 30)
        .map(String::from)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_lowercases_and_drops_short_tokens() {
        let tokens = tokenize("XDNA2 NPU, a AIE-tiles !");
        assert_eq!(tokens, vec!["xdna2", "npu", "aie", "tiles"]);
    }

    #[test]
    fn query_ranks_relevant_doc_first() {
        let mut idx = Bm25Index::new();
        idx.add("Le NPU XDNA2 execute des LLM sur Ryzen AI", "doc1.md");
        idx.add("Une recette de gateau au chocolat", "doc2.md");
        idx.add("AIE tiles sont les unites de calcul du NPU XDNA2", "doc3.md");

        let results = idx.query("NPU XDNA2", 3);
        assert!(!results.is_empty());
        assert_ne!(results[0].source, "doc2.md", "le doc hors-sujet ne doit pas etre premier");
    }

    #[test]
    fn query_on_empty_index_returns_nothing() {
        let mut idx = Bm25Index::new();
        assert!(idx.query("quoi que ce soit", 5).is_empty());
    }

    #[test]
    fn save_then_load_roundtrip_preserves_query_results() {
        let mut idx = Bm25Index::new();
        idx.add("Le NPU XDNA2 execute des LLM", "doc1.md");
        idx.add("AIE tiles et XRT runtime", "doc2.md");

        let path = std::env::temp_dir().join(format!("jarvis_mini_bm25_test_{}.json", std::process::id()));
        let path_str = path.to_str().unwrap();

        idx.save(path_str).unwrap();
        let mut reloaded = Bm25Index::load(path_str).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(reloaded.len(), 2);
        let results = reloaded.query("XDNA2", 5);
        assert_eq!(results[0].source, "doc1.md");
    }
}
