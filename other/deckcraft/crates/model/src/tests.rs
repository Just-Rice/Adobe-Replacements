use super::*;
use crate::resolve::Ctx;

#[test]
fn new_presentation_is_valid() {
    let p = Presentation::default();
    assert_eq!(p.slides.len(), 1);
    assert_eq!(p.masters.len(), 1);
    assert_eq!(p.masters[0].layouts.len(), 11);
    assert!(p.validate().is_empty(), "{:?}", p.validate());
    let s = &p.slides[0];
    let (m, l) = p.master_for(s).unwrap();
    assert_eq!(l.unwrap().kind, LayoutType::Title);
    assert_eq!(m.theme.name, "DeckCraft");
    // Title slide has a centred title and a subtitle.
    let kinds: Vec<_> = s.shapes.iter().filter_map(|x| x.ph_type()).collect();
    assert_eq!(kinds, vec![PhType::CtrTitle, PhType::SubTitle]);
}

#[test]
fn placeholder_inherits_position_and_styles() {
    let p = Presentation::default();
    let s = &p.slides[0];
    let ctx = Ctx::for_slide(&p, s).unwrap();
    let title = &s.shapes[0];
    let x = resolve::xfrm(&ctx, title);
    assert!((x.x - 120.0).abs() < 1e-9 && (x.w - 720.0).abs() < 1e-9);
    let lvl = resolve::level_style(&ctx, title, 0);
    assert_eq!(lvl.run.size, Some(60.0)); // layout overrides the master's 44
    assert_eq!(lvl.para.align, Some(text::Align::Center));
    assert_eq!(resolve::font_family(&ctx, &lvl.run), "Inter");
    let body = resolve::body(&ctx, title);
    assert_eq!(body.anchor, Some(text::Anchor::Bottom));
}

#[test]
fn body_levels_from_master() {
    let mut p = Presentation::default();
    let lay = defaults::layout_of_kind(&p, LayoutType::TitleAndContent).unwrap();
    let s = defaults::new_slide(&mut p, lay);
    let ctx = Ctx::for_slide(&p, &s).unwrap();
    let body = &s.shapes[1];
    assert_eq!(body.ph_type(), Some(PhType::Obj));
    let l0 = resolve::level_style(&ctx, body, 0);
    let l1 = resolve::level_style(&ctx, body, 1);
    assert_eq!(l0.run.size, Some(28.0));
    assert_eq!(l1.run.size, Some(24.0));
    assert!(matches!(l0.para.bullet, Some(text::Bullet::Char { .. })));
}

#[test]
fn shape_style_fill_resolves_through_theme() {
    let p = Presentation::default();
    let s = &p.slides[0];
    let ctx = Ctx::for_slide(&p, s).unwrap();
    let sh = Shape { style: Some(ShapeStyle::accent(SchemeSlot::Accent1)), xfrm: Some(Xfrm::new(0.0, 0.0, 10.0, 10.0)), ..Default::default() };
    let (f, ph) = resolve::fill(&ctx, &sh);
    let Some(Fill::Solid { color }) = f else { panic!("expected solid") };
    assert_eq!(ctx.color(&color, ph), p.masters[0].theme.colors.get(SchemeSlot::Accent1));
    let (l, ph) = resolve::line(&ctx, &sh);
    assert!(l.width.is_some());
    assert!(ph.is_some());
}

#[test]
fn serde_roundtrip() {
    let p = Presentation::default();
    let j = serde_json::to_string(&p).unwrap();
    let q: Presentation = serde_json::from_str(&j).unwrap();
    assert_eq!(p, q);
    // Unknown and missing fields are tolerated.
    let q: Presentation = serde_json::from_str(r#"{"slides":[],"zzz":1}"#).unwrap();
    assert!(q.slides.is_empty());
}

#[test]
fn ids_are_unique_and_fix_next_id() {
    let mut p = Presentation::default();
    let a = p.alloc_id();
    let b = p.alloc_id();
    assert!(b > a);
    p.next_id = 0;
    p.fix_next_id();
    assert!(p.next_id >= b.min(p.next_id));
    let c = p.alloc_id();
    let mut all = vec![];
    for s in &p.slides {
        all.push(s.id.0);
        walk(&s.shapes, &mut |sh, _| all.push(sh.id.0));
    }
    assert!(!all.contains(&c));
}

#[test]
fn group_find_and_walk() {
    let child = Shape { id: ShapeId(900), ..Default::default() };
    let g = Shape { id: ShapeId(901), kind: ShapeKind::Group { children: vec![child], child: Xfrm::new(0.0, 0.0, 1.0, 1.0) }, ..Default::default() };
    let mut slide = Slide { shapes: vec![g], ..Default::default() };
    assert!(slide.shape(ShapeId(900)).is_some());
    assert!(slide.shape_mut(ShapeId(900)).is_some());
    let mut n = 0;
    walk(&slide.shapes, &mut |_, _| n += 1);
    assert_eq!(n, 2);
}

#[test]
fn text_body_helpers() {
    let mut t = TextBody::from_text("Hello\nWorld");
    assert_eq!(t.text(), "Hello\nWorld");
    assert_eq!(t.char_len(), 11);
    t.paragraphs[0].runs.push(Run::new(" there"));
    t.paragraphs[0].normalize();
    assert_eq!(t.paragraphs[0].runs.len(), 1);
    assert_eq!(t.paragraphs[0].text(), "Hello there");
}

#[test]
fn builtin_themes_have_distinct_names() {
    let t = theme::builtin_themes();
    let mut names: Vec<_> = t.iter().map(|t| t.name.clone()).collect();
    names.sort();
    names.dedup();
    assert_eq!(names.len(), t.len());
    assert!(t.len() >= 8);
}

#[test]
fn four_by_three_master_scales() {
    let p = defaults::blank_presentation(defaults::STANDARD, Theme::default(), true);
    let m = &p.masters[0];
    let title = m.shapes.iter().find(|s| s.ph_type() == Some(PhType::Title)).unwrap();
    let x = title.xfrm.unwrap();
    assert!(x.x + x.w <= 720.0 + 1e-6);
}
