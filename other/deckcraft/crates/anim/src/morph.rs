//! Morph: matching shapes between two slides and interpolating their boxes.

use deckcraft_model::{Shape, ShapeId, Slide, Xfrm};

use crate::clampf;

fn text_of(s: &Shape) -> String {
    s.text.as_ref().map(|t| t.text()).unwrap_or_default()
}

/// Pairs of (shape on `a`, shape on `b`) that Morph animates from one to the other: first by
/// `!!name` (forced match), then by exact name, then by the same kind and text, then by the same
/// kind and preset geometry. Each shape is used once; top-level shapes only (groups move whole).
pub fn morph_pairs(a: &Slide, b: &Slide) -> Vec<(ShapeId, ShapeId)> {
    let mut pairs: Vec<(ShapeId, ShapeId)> = Vec::new();
    let mut used_a = vec![false; a.shapes.len()];
    let mut used_b = vec![false; b.shapes.len()];
    let passes: [&dyn Fn(&Shape, &Shape) -> bool; 4] = [
        &|x, y| x.name.starts_with("!!") && x.name == y.name,
        &|x, y| !x.name.is_empty() && x.name == y.name,
        &|x, y| {
            if x.kind_name() != y.kind_name() {
                return false;
            }
            let tx = text_of(x);
            !tx.trim().is_empty() && tx == text_of(y)
        },
        &|x, y| {
            x.kind_name() == y.kind_name()
                && x.geom.preset_name().is_some()
                && x.geom.preset_name() == y.geom.preset_name()
                && text_of(x).trim().is_empty() == text_of(y).trim().is_empty()
        },
    ];
    for pass in passes {
        for (i, sa) in a.shapes.iter().enumerate() {
            if used_a.get(i).copied().unwrap_or(true) {
                continue;
            }
            let found = b.shapes.iter().enumerate().find(|(j, sb)| !used_b.get(*j).copied().unwrap_or(true) && pass(sa, sb));
            if let Some((j, sb)) = found {
                if let Some(u) = used_a.get_mut(i) {
                    *u = true;
                }
                if let Some(u) = used_b.get_mut(j) {
                    *u = true;
                }
                pairs.push((sa.id, sb.id));
            }
        }
    }
    pairs
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    let v = a + (b - a) * t;
    if v.is_finite() {
        v
    } else if t < 0.5 {
        a
    } else {
        b
    }
}

/// Interpolate a box from `a` to `b` at `t` (0..1): position and size linearly, rotation along the
/// shortest way round; flips switch at the midpoint.
pub fn morph_xfrm(a: Xfrm, b: Xfrm, t: f64) -> Xfrm {
    let t = clampf(t, 0.0, 1.0, 0.0);
    if t >= 1.0 {
        return b;
    }
    if t <= 0.0 {
        return a;
    }
    let ra = if a.rot.is_finite() { a.rot } else { 0.0 };
    let rb = if b.rot.is_finite() { b.rot } else { 0.0 };
    let mut d = (rb - ra).rem_euclid(360.0);
    if d > 180.0 {
        d -= 360.0;
    }
    let rot = (ra + d * t).rem_euclid(360.0);
    let late = t >= 0.5;
    Xfrm {
        x: lerp(a.x, b.x, t),
        y: lerp(a.y, b.y, t),
        w: lerp(a.w, b.w, t),
        h: lerp(a.h, b.h, t),
        rot: if rot.is_finite() { rot } else { 0.0 },
        flip_h: if late { b.flip_h } else { a.flip_h },
        flip_v: if late { b.flip_v } else { a.flip_v },
    }
}
