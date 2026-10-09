#[derive(Copy, Clone, Debug)]
pub enum Color {
  Black,
  White,
  Red,
  Green,
  Blue,
  Custom { r: f32, g: f32, b: f32 },
}

impl Color {
  pub fn get_f32_r(&self) -> f32 {
    match &self {
      Color::Black => 0.0f32,
      Color::White => 1.0f32,
      Color::Red => 1.0f32,
      Color::Green => 0.0f32,
      Color::Blue => 0.0f32,
      Color::Custom {r, g, b} => *r,
    }
  }

  pub fn get_f32_g(&self) -> f32 {
    match &self {
      Color::Black => 0.0f32,
      Color::White => 1.0f32,
      Color::Red => 0.0f32,
      Color::Green => 1.0f32,
      Color::Blue => 0.0f32,
      Color::Custom {r, g, b} => *g,
    }
  }

  pub fn get_f32_b(&self) -> f32 {
    match &self {
      Color::Black => 0.0f32,
      Color::White => 1.0f32,
      Color::Red => 0.0f32,
      Color::Green => 0.0f32,
      Color::Blue => 1.0f32,
      Color::Custom {r, g, b} => *b,
    }
  }


  pub fn get_u32_r(&self) -> u32 {
    match &self {
      Color::Black => 0,
      Color::White => 255,
      Color::Red => 255,
      Color::Green => 0,
      Color::Blue => 0,
      Color::Custom {r, g, b} => *r as u32,
    }
  }

  pub fn get_u32_g(&self) -> u32 {
    match &self {
      Color::Black => 0,
      Color::White => 255,
      Color::Red => 0,
      Color::Green => 255,
      Color::Blue => 0,
      Color::Custom {r, g, b} => *g as u32,
    }
  }

  pub fn get_u32_b(&self) -> u32 {
    match &self {
      Color::Black => 0,
      Color::White => 255,
      Color::Red => 0,
      Color::Green => 0,
      Color::Blue => 255,
      Color::Custom {r, g, b} => *b as u32,
    }
  }

}
