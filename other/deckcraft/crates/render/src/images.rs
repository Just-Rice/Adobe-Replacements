//! Decoded pictures shared by every render on this process, keyed by the media bytes' address and
//! the requested size level and adjustments, so each picture is decoded once per size.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use deckcraft_model::style::PictureAdjust;
use vello_cpu::Pixmap;

const BUDGET: usize = 512 << 20;

struct Entry {
    _bytes: Arc<Vec<u8>>,
    pm: Arc<Pixmap>,
    stamp: u64,
}

#[derive(Default)]
struct Cache {
    map: HashMap<(usize, u32, u64), Entry>,
    bytes: usize,
    clock: u64,
    bad: HashMap<usize, Arc<Vec<u8>>>,
}

static CACHE: Mutex<Option<Cache>> = Mutex::new(None);

fn with<R>(f: impl FnOnce(&mut Cache) -> R) -> R {
    let mut g = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(Cache::default))
}

fn adjust_key(a: &PictureAdjust) -> u64 {
    let mut h: u64 = 1469598103934665603;
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(1099511628211);
    };
    mix(a.brightness.to_bits());
    mix(a.contrast.to_bits());
    mix(a.saturation.unwrap_or(1.0).to_bits());
    mix(a.grayscale as u64);
    mix(a.sharpen.to_bits());
    if let Some(c) = a.clear_color {
        mix(((c.r as u64) << 16) | ((c.g as u64) << 8) | c.b as u64 | 1 << 40);
    }
    h
}

/// Decode encoded bytes (PNG, JPEG, GIF, WebP, BMP, TIFF) to straight RGBA.
pub fn decode(bytes: &[u8]) -> Option<image::RgbaImage> {
    if bytes.len() > 512 << 20 {
        return None;
    }
    let img = image::load_from_memory(bytes).ok()?;
    let (w, h) = (img.width(), img.height());
    if w == 0 || h == 0 || (w as u64) * (h as u64) > 400_000_000 {
        return None;
    }
    Some(img.to_rgba8())
}

/// A premultiplied pixmap of the picture, downsampled so its longer side is near `want` px.
pub fn decode_for(bytes: &Arc<Vec<u8>>, want: f64, adj: &PictureAdjust) -> Option<Arc<Pixmap>> {
    let addr = Arc::as_ptr(bytes) as usize;
    if with(|c| c.bad.contains_key(&addr)) {
        return None;
    }
    // Power-of-two size buckets so zooming doesn't re-decode every frame.
    let bucket = (want.max(16.0).log2().ceil() as u32).min(15);
    let key = (addr, bucket, adjust_key(adj));
    if let Some(pm) = with(|c| {
        c.clock += 1;
        let clock = c.clock;
        c.map.get_mut(&key).map(|e| {
            e.stamp = clock;
            e.pm.clone()
        })
    }) {
        return Some(pm);
    }
    let Some(mut img) = decode(bytes) else {
        with(|c| {
            c.bad.insert(addr, bytes.clone());
        });
        return None;
    };
    let target = 1u32 << bucket;
    let (w, h) = img.dimensions();
    if w.max(h) > target.max(16) && w.max(h) > 4096.min(target * 2) {
        let k = target as f64 / w.max(h) as f64;
        let (nw, nh) = (((w as f64 * k).round() as u32).max(1), ((h as f64 * k).round() as u32).max(1));
        img = image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Triangle);
    }
    apply_adjust(&mut img, adj);
    let (w, h) = img.dimensions();
    if w > u16::MAX as u32 || h > u16::MAX as u32 {
        return None;
    }
    let data: Vec<vello_cpu::color::PremulRgba8> = img
        .pixels()
        .map(|p| {
            let a = p[3] as u16;
            let m = |c: u8| ((c as u16 * a + 127) / 255) as u8;
            vello_cpu::color::PremulRgba8 { r: m(p[0]), g: m(p[1]), b: m(p[2]), a: p[3] }
        })
        .collect();
    let pm = Arc::new(Pixmap::from_parts(data, w as u16, h as u16));
    with(|c| {
        c.clock += 1;
        let n = w as usize * h as usize * 4;
        c.bytes += n;
        c.map.insert(key, Entry { _bytes: bytes.clone(), pm: pm.clone(), stamp: c.clock });
        while c.bytes > BUDGET && c.map.len() > 1 {
            let Some(oldest) = c.map.iter().min_by_key(|(_, e)| e.stamp).map(|(k, _)| *k) else { break };
            if let Some(e) = c.map.remove(&oldest) {
                c.bytes = c.bytes.saturating_sub(e.pm.width() as usize * e.pm.height() as usize * 4);
            }
        }
    });
    Some(pm)
}

fn apply_adjust(img: &mut image::RgbaImage, a: &PictureAdjust) {
    let sat = a.saturation.unwrap_or(1.0);
    if a.brightness == 0.0 && a.contrast == 0.0 && sat == 1.0 && !a.grayscale && a.clear_color.is_none() && a.duotone.is_none() {
        return;
    }
    let b = a.brightness.clamp(-1.0, 1.0) * 255.0;
    let c = a.contrast.clamp(-1.0, 1.0);
    let cf = if c >= 0.0 { 1.0 / (1.0 - c * 0.99) } else { 1.0 + c };
    for p in img.pixels_mut() {
        if let Some(cc) = a.clear_color
            && (p[0] as i32 - cc.r as i32).abs() < 8
            && (p[1] as i32 - cc.g as i32).abs() < 8
            && (p[2] as i32 - cc.b as i32).abs() < 8
        {
            p[3] = 0;
            continue;
        }
        let mut rgb = [p[0] as f64, p[1] as f64, p[2] as f64];
        let y = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
        let s = if a.grayscale { 0.0 } else { sat };
        for v in &mut rgb {
            *v = y + (*v - y) * s;
            *v = (*v - 128.0) * cf + 128.0 + b;
        }
        for (i, v) in rgb.iter().enumerate() {
            if let Some(ch) = p.0.get_mut(i) {
                *ch = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}
