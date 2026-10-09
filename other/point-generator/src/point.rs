use bytes::{BytesMut, BufMut};
use rand::distributions::Uniform;
use rand::Rng;
use crate::color::Color;
use byteorder::{WriteBytesExt, LittleEndian, BigEndian};


pub struct Point {
  pub x: f32,
  pub y: f32,
  pub z: f32,
  _unused: f32,
  pub r: f32,
  pub g: f32,
  pub b: f32,
  pub a: f32,
}

impl Point {

  pub fn at(x: f32, y: f32, z: f32, color: Color) -> Self {
    Point {
      x,
      y,
      z,
      _unused: 0.0f32,
      r: color.get_f32_r(),
      g: color.get_f32_g(),
      b: color.get_f32_b(),
      a: 1.0f32,
    }
  }

  pub fn at_random_range(min: f32, max: f32, color: Color) -> Self {
    let mut rng = rand::thread_rng();
    let range = Uniform::new_inclusive(min, max);

    let x = rng.sample(&range);
    let y = rng.sample(&range);
    let z = rng.sample(&range);

    Point {
      x,
      y,
      z,
      _unused: 0.0f32,
      r: color.get_f32_r(),
      g: color.get_f32_g(),
      b: color.get_f32_b(),
      a: 1.0f32,
    }
  }


  pub fn to_bytes(&self) -> Vec<u8> {
    let mut buf = Vec::with_capacity(32);
    buf.write_f32::<LittleEndian>(self.x);
    buf.write_f32::<LittleEndian>(self.y);
    buf.write_f32::<LittleEndian>(self.z);
    buf.write_f32::<LittleEndian>(0.0f32);
    buf.write_f32::<LittleEndian>(self.r);
    buf.write_f32::<LittleEndian>(self.g);
    buf.write_f32::<LittleEndian>(self.b);
    buf.write_f32::<LittleEndian>(self.a);
    buf
  }

  pub fn location_string(&self) -> String {
    format!("{}, {}, {}", self.x, self.y, self.z)
  }

  pub fn debug_string(&self) -> String {
    format!("(location: {}, {}, {}); (color: {}, {}, {}, alpha: {})",
            self.x, self.y, self.z,
            self.r, self.g, self.b, self.a)
  }
}