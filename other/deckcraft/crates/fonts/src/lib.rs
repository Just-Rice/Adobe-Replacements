//! DeckCraft fonts: the font database (craft-fonts families, user and system fonts, and open
//! substitutes for the Office fonts presentations ask for), vertical metrics, glyph outlines and
//! OpenType shaping.
//!
//! Shaping here is style-agnostic: [`shape`] turns a string in one face into glyph ids, clusters
//! and advances in font units. `deckcraft-text` applies sizes, tracking, scaling and
//! justification on top.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]
#![forbid(unsafe_code)]

mod fontdb;

pub use fontdb::{FALLBACK_FAMILY, FaceRef, FontDb, FontFace, LAST_RESORT_FAMILY, base_style, bundled, substitutes, system_font_dirs};
pub use harfrust::Feature;
use harfrust::{Direction, ShapeOptions, Tag, UnicodeBuffer};
pub use kurbo::BezPath;
use skrifa::MetadataProvider;
use skrifa::instance::Size;

/// A font from the optional craft-fonts build input (https://github.com/storytold/craft-fonts;
/// empty unless built with `CRAFT_FONTS_DIR`, see `build.rs`).
pub struct CraftFont {
    pub family: &'static str,
    pub style: &'static str,
    /// ISO 15924 scripts the font is for, e.g. `"Jpan"`.
    pub scripts: &'static [&'static str],
    pub bytes: &'static [u8],
}

include!(concat!(env!("OUT_DIR"), "/craft_fonts.rs"));

/// The craft-fonts faces for Japanese (`"Jpan"`), in manifest order. Empty without craft-fonts.
pub fn japanese_fonts() -> impl Iterator<Item = &'static CraftFont> {
    CRAFT_FONTS.iter().filter(|f| f.scripts.contains(&"Jpan"))
}

/// Japanese faces in document-fallback order: Mincho (serif) families first, matching the
/// serif default text font, then the others; Regular before other styles.
pub fn japanese_document_fonts() -> Vec<&'static CraftFont> {
    let mut v: Vec<_> = japanese_fonts().collect();
    v.sort_by_key(|f| (!f.family.contains("Mincho"), f.style != "Regular"));
    v
}

/// Japanese faces in UI order: BIZ UDPGothic first (the UI face), then the others; `bold` puts
/// bold styles before regular ones.
pub fn japanese_ui_fonts(bold: bool) -> Vec<&'static CraftFont> {
    let mut v: Vec<_> = japanese_fonts().collect();
    v.sort_by_key(|f| (f.family != "BIZ UDPGothic", (f.style == "Bold") != bold));
    v
}

/// The default theme's font.
pub const DEFAULT_FAMILY: &str = "Inter";

/// One shaped glyph, in font units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapedGlyph {
    pub gid: u32,
    /// Byte offset (in the shaped string) of the cluster this glyph belongs to.
    pub cluster: usize,
    pub x_advance: i32,
    pub x_offset: i32,
    pub y_offset: i32,
}

/// An OpenType feature setting: `"liga"`, `"-kern"`, `"ss01"`.
pub fn feature(tag: &str) -> Option<Feature> {
    let (on, t) = match tag.strip_prefix('-') {
        Some(r) => (false, r),
        None => (true, tag.strip_prefix('+').unwrap_or(tag)),
    };
    let b = t.as_bytes();
    if b.len() != 4 {
        return None;
    }
    Some(Feature::new(Tag::new(&[b[0], b[1], b[2], b[3]]), on as u32, ..))
}

/// A strong right-to-left character (Hebrew, Arabic, …)?
pub fn is_rtl(c: char) -> bool {
    use unicode_bidi::BidiClass::{AL, R};
    matches!(unicode_bidi::bidi_class(c), R | AL)
}

/// Maximal runs of one direction (neutrals join the run they're in): (byte range, right-to-left).
fn direction_runs(text: &str) -> Vec<(std::ops::Range<usize>, bool)> {
    let mut runs: Vec<(std::ops::Range<usize>, bool)> = Vec::new();
    let mut cur: Option<bool> = None;
    let mut start = 0;
    for (i, c) in text.char_indices() {
        let strong = if is_rtl(c) {
            Some(true)
        } else if unicode_bidi::bidi_class(c) == unicode_bidi::BidiClass::L {
            Some(false)
        } else {
            None
        };
        match (cur, strong) {
            (None, Some(d)) => cur = Some(d),
            (Some(a), Some(d)) if a != d => {
                runs.push((start..i, a));
                start = i;
                cur = Some(d);
            }
            _ => {}
        }
    }
    runs.push((start..text.len(), cur.unwrap_or(false)));
    runs
}

/// Shape `text` with `face`. `chars` lets callers substitute characters (e.g. uppercase for All
/// Caps) while keeping clusters pointing into the original string. Glyphs come in logical
/// order: right-to-left runs are shaped right to left, then their clusters put back in text
/// order (each cluster's glyphs keep the shaper's order); line layout reorders them visually.
pub fn shape(face: &FontFace, text: &str, features: &[Feature], map: impl Fn(char) -> char) -> Vec<ShapedGlyph> {
    if !text.chars().any(is_rtl) {
        return shape_dir(face, text, features, &map, false);
    }
    let mut out = Vec::with_capacity(text.len());
    for (r, rtl) in direction_runs(text) {
        let mut g = shape_dir(face, &text[r.clone()], features, &map, rtl);
        for x in &mut g {
            x.cluster += r.start;
        }
        if rtl {
            // Visual (clusters descending) → logical, cluster by cluster.
            let mut groups: Vec<Vec<ShapedGlyph>> = Vec::new();
            for x in g {
                match groups.last_mut() {
                    Some(last) if last[0].cluster == x.cluster => last.push(x),
                    _ => groups.push(vec![x]),
                }
            }
            groups.sort_by_key(|grp| grp[0].cluster);
            g = groups.into_iter().flatten().collect();
        }
        out.extend(g);
    }
    out
}

fn shape_dir(face: &FontFace, text: &str, features: &[Feature], map: &impl Fn(char) -> char, rtl: bool) -> Vec<ShapedGlyph> {
    let mut out = Vec::with_capacity(text.len());
    let shaped = face.hb().map(|hb| {
        let shaper = face.shaper.shaper(&hb).instance(face.instance.as_ref()).build();
        let mut buf = UnicodeBuffer::new();
        for (i, c) in text.char_indices() {
            buf.add(map(c), i as u32);
        }
        buf.guess_segment_properties();
        buf.set_direction(if rtl { Direction::RightToLeft } else { Direction::LeftToRight });
        let gb = shaper.shape(buf, ShapeOptions::new().features(features));
        for (info, pos) in gb.glyph_infos().iter().zip(gb.glyph_positions()) {
            out.push(ShapedGlyph {
                gid: info.glyph_id,
                cluster: info.cluster as usize,
                x_advance: pos.x_advance,
                x_offset: pos.x_offset,
                y_offset: pos.y_offset,
            });
        }
    });
    if shaped.is_none()
        && let Some(f) = face.skrifa()
    {
        let cmap = f.charmap();
        let gm = f.glyph_metrics(Size::unscaled(), face.location());
        for (i, c) in text.char_indices() {
            let g = cmap.map(map(c)).unwrap_or_default();
            let adv = gm.advance_width(g).unwrap_or(face.upem as f32 * 0.5);
            out.push(ShapedGlyph { gid: g.to_u32(), cluster: i, x_advance: adv.round() as i32, x_offset: 0, y_offset: 0 });
        }
    }
    out
}

/// Glyph id of the first of `chars` the face has (0 = .notdef).
pub fn first_glyph(face: &FontFace, chars: &[char]) -> u32 {
    chars.iter().map(|c| face.glyph_for(*c)).find(|g| *g != 0).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn any_face() -> std::sync::Arc<FontFace> {
        FontDb::global().face(DEFAULT_FAMILY, "Regular")
    }

    #[test]
    fn last_resort_face_parses() {
        let f = fontdb::last_resort_face();
        assert_eq!(f.family, fontdb::LAST_RESORT_FAMILY);
        assert!(f.upem > 0.0);
    }

    #[test]
    fn some_face_always_resolves() {
        let db = FontDb::with_font_dirs(Vec::new());
        db.set_system_fallback(false);
        let f = db.face("No Such Font Anywhere", "Regular");
        assert!(!f.family.is_empty());
        assert!(!shape(&f, "Hello", &[], |c| c).is_empty());
    }

    #[test]
    fn office_fonts_have_substitutes() {
        assert!(substitutes("Calibri").contains(&"Carlito"));
        assert!(substitutes("Cambria").contains(&"Caladea"));
        assert!(substitutes("Times New Roman").contains(&"Liberation Serif"));
        assert!(substitutes("Aptos").contains(&"Inter"));
        assert!(substitutes("zzz").is_empty());
    }

    #[test]
    fn shaping_produces_clusters_and_advances() {
        let face = any_face();
        let g = shape(&face, "Hello", &[], |c| c);
        assert_eq!(g.len(), 5);
        assert_eq!(g.iter().map(|g| g.cluster).collect::<Vec<_>>(), vec![0, 1, 2, 3, 4]);
        assert!(g.iter().all(|g| g.x_advance > 0 && g.gid != 0));
    }

    #[test]
    fn mapping_keeps_clusters() {
        let face = any_face();
        let g = shape(&face, "ab", &[], |c| c.to_ascii_uppercase());
        assert_eq!(g[0].gid, face.glyph_for('A'));
        assert_eq!(g[1].cluster, 1);
    }

    #[test]
    fn outlines_and_metrics() {
        let db = FontDb::global();
        let face = any_face();
        let o = db.outline(&face, face.glyph_for('O'));
        assert!(!o.elements().is_empty());
        assert!(face.ascent > 0.0 && face.descent > 0.0);
        assert!(feature("abc").is_none());
        assert!(feature("liga").is_some());
    }

    #[test]
    fn craft_fonts_latin_when_present() {
        if !CRAFT_FONTS.iter().any(|f| f.family == "Inter") {
            eprintln!("skipped: built without craft-fonts' Latin manifest (set CRAFT_FONTS_DIR)");
            return;
        }
        let db = FontDb::with_font_dirs(Vec::new());
        db.set_system_fallback(false);
        assert_eq!(db.face("Inter", "Regular").family, "Inter");
        assert_eq!(db.face("Aptos", "Regular").family, "Inter");
        if CRAFT_FONTS.iter().any(|f| f.family == "Carlito") {
            assert_eq!(db.face("Calibri", "Bold").family, "Carlito");
        }
    }
}
