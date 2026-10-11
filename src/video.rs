//! Ingestion video — porte en Rust la LEÇON tiree du debug torchcodec/ffmpeg
//! (voir conversation) : un seul process ffmpeg pour extraire un lot de frames,
//! jamais un process par frame (c'etait la cause des ~5.6 fps mesures a l'origine ;
//! decoder tout le lot en un seul appel donnait ~500 fps avec la meme commande).
//!
//! Honnetete deliberee sur GPU vs CPU :
//! Le decodage materiel (NVDEC, via `-hwaccel cuda`) peut etre demande, mais les
//! octets qui reviennent ici passent TOUJOURS par un pipe stdout (memoire hote).
//! Ce module ne pretend jamais que le buffer retourne est "deja sur le GPU" —
//! c'est exactement l'erreur trouvee a plusieurs reprises dans les scripts Python
//! examines (`ffmpeg_nvdec_decoder.py`, `FFmpegCUDADecoder.decode_all()`, et le
//! snippet "from nvidia import cuda" qui ne correspond a aucune API reelle).
//! Si un jour ce module sert un pipeline GPU, le transfert `.cuda()`/upload doit
//! etre fait explicitement par l'appelant, sur ce buffer CPU.

use serde::Deserialize;
use std::io::Read;
use std::process::{Command, Stdio};

#[derive(Debug, Clone)]
pub struct VideoMetadata {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub duration_s: f64,
    pub codec: String,
}

#[derive(Debug, Deserialize)]
struct FfprobeOutput {
    streams: Vec<FfprobeStream>,
    format: FfprobeFormat,
}

#[derive(Debug, Deserialize)]
struct FfprobeStream {
    codec_type: String,
    #[serde(default)]
    codec_name: String,
    #[serde(default)]
    width: u32,
    #[serde(default)]
    height: u32,
    #[serde(default)]
    r_frame_rate: String,
}

#[derive(Debug, Deserialize)]
struct FfprobeFormat {
    #[serde(default)]
    duration: String,
}

/// Un decodeur video minimal, base sur des appels subprocess a `ffmpeg`/`ffprobe`.
/// Necessite les deux binaires dans le PATH.
pub struct VideoSource {
    pub path: String,
    pub meta: VideoMetadata,
    ffmpeg_bin: String,
}

/// Une frame decodee, au format RGB24 planaire simple (H, W, 3), en memoire hote (CPU).
pub struct Frame {
    pub index: usize,
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

impl VideoSource {
    pub fn open(path: &str) -> anyhow::Result<Self> {
        Self::open_with_bins(path, "ffmpeg", "ffprobe")
    }

    pub fn open_with_bins(path: &str, ffmpeg_bin: &str, ffprobe_bin: &str) -> anyhow::Result<Self> {
        if !std::path::Path::new(path).exists() {
            anyhow::bail!("video introuvable: {path}");
        }
        let meta = probe(path, ffprobe_bin)?;
        Ok(Self {
            path: path.to_string(),
            meta,
            ffmpeg_bin: ffmpeg_bin.to_string(),
        })
    }

    /// Extrait `count` frames consecutives a partir de la frame approx. `start_frame`.
    ///
    /// UN SEUL appel ffmpeg pour tout le lot (c'est la partie qui compte). Le seek
    /// utilise `-ss` AVANT `-i` : rapide, mais precis seulement a la keyframe la
    /// plus proche (limite documentee, cf. conversation sur ffmpeg_nvdec_decoder.py).
    /// Pour un usage "extraire une poignee de frames representatives d'une video
    /// pour un pipeline RAG multimodal", c'est un compromis raisonnable ; ce n'est
    /// pas un acces frame-exact.
    pub fn extract_frames(&self, start_frame: usize, count: usize, hwaccel: bool) -> anyhow::Result<Vec<Frame>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let fps = if self.meta.fps > 0.0 { self.meta.fps } else { 25.0 };
        let timestamp = start_frame as f64 / fps;

        let mut cmd = Command::new(&self.ffmpeg_bin);
        cmd.arg("-y").arg("-v").arg("error");
        cmd.arg("-ss").arg(format!("{timestamp:.6}"));
        if hwaccel {
            // Decode materiel si dispo (NVDEC). Le resultat repasse quand meme par
            // stdout (CPU) plus bas -- voir note d'honnetete en tete de fichier.
            cmd.arg("-hwaccel").arg("cuda");
        }
        cmd.arg("-i").arg(&self.path);
        cmd.arg("-frames:v").arg(count.to_string());
        cmd.arg("-pix_fmt").arg("rgb24");
        cmd.arg("-f").arg("rawvideo");
        cmd.arg("-");
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let output = cmd.output()?;
        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("ffmpeg a echoue: {err}");
        }

        let w = self.meta.width as usize;
        let h = self.meta.height as usize;
        let frame_size = w * h * 3;
        if frame_size == 0 {
            anyhow::bail!("resolution video inconnue (width/height=0)");
        }

        let raw = output.stdout;
        let n_complete = raw.len() / frame_size;
        if n_complete == 0 {
            anyhow::bail!(
                "aucune frame complete recue ({} octets recus, {} attendus par frame)",
                raw.len(),
                frame_size
            );
        }

        let mut frames = Vec::with_capacity(n_complete);
        for i in 0..n_complete {
            let start = i * frame_size;
            let end = start + frame_size;
            frames.push(Frame {
                index: start_frame + i,
                width: self.meta.width,
                height: self.meta.height,
                rgb: raw[start..end].to_vec(),
            });
        }
        Ok(frames)
    }
}

impl Frame {
    /// Ecrit la frame en PPM (P6) — format trivial, lisible par la plupart des
    /// visionneuses/convertisseurs, sans dependance a une lib image.
    pub fn save_ppm(&self, path: &str) -> anyhow::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        write!(f, "P6\n{} {}\n255\n", self.width, self.height)?;
        f.write_all(&self.rgb)?;
        Ok(())
    }
}

fn probe(path: &str, ffprobe_bin: &str) -> anyhow::Result<VideoMetadata> {
    let output = Command::new(ffprobe_bin)
        .arg("-v").arg("error")
        .arg("-print_format").arg("json")
        .arg("-show_format")
        .arg("-show_streams")
        .arg(path)
        .output()?;

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ffprobe a echoue: {err}");
    }

    let mut stdout = String::new();
    std::io::Cursor::new(output.stdout).read_to_string(&mut stdout)?;
    let parsed: FfprobeOutput = serde_json::from_str(&stdout)?;

    let video_stream = parsed
        .streams
        .into_iter()
        .find(|s| s.codec_type == "video")
        .ok_or_else(|| anyhow::anyhow!("aucun flux video trouve"))?;

    let fps = parse_frame_rate(&video_stream.r_frame_rate);
    let duration_s: f64 = parsed.format.duration.parse().unwrap_or(0.0);

    Ok(VideoMetadata {
        width: video_stream.width,
        height: video_stream.height,
        fps,
        duration_s,
        codec: video_stream.codec_name,
    })
}

fn parse_frame_rate(s: &str) -> f64 {
    // Format ffprobe habituel: "30/1" ou "30000/1001"
    if let Some((num, den)) = s.split_once('/') {
        let n: f64 = num.parse().unwrap_or(0.0);
        let d: f64 = den.parse().unwrap_or(1.0);
        if d != 0.0 {
            return n / d;
        }
    }
    s.parse().unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_integer_frame_rate() {
        assert_eq!(parse_frame_rate("30/1"), 30.0);
    }

    #[test]
    fn parses_ntsc_frame_rate() {
        let fps = parse_frame_rate("30000/1001");
        assert!((fps - 29.97).abs() < 0.01);
    }

    #[test]
    fn parses_plain_number_without_slash() {
        assert_eq!(parse_frame_rate("25"), 25.0);
    }

    #[test]
    fn falls_back_to_zero_on_garbage() {
        assert_eq!(parse_frame_rate("n/a"), 0.0);
    }

    #[test]
    fn open_fails_cleanly_on_missing_file() {
        let result = VideoSource::open("/chemin/qui/n_existe_pas.mp4");
        assert!(result.is_err(), "doit renvoyer une erreur, pas paniquer");
    }
}
