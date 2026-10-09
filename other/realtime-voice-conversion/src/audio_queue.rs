use zmq;
use zmq::{Error, Socket};
use std::collections::VecDeque;
use std::sync::{RwLock, PoisonError};

pub struct AudioQueue {
  queue: RwLock<VecDeque<f32>>,
}

impl AudioQueue {
  pub fn new() -> Self {
    Self {
      queue: RwLock::new(VecDeque::new()),
    }
  }

  pub fn push_back(&self, val: f32) {
    match self.queue.write() {
      Ok(mut queue) => {
        queue.push_back(val);
      },
      Err(_) => {
        unreachable!("This shouldn't happen (push_back)");
      },
    }
  }

  pub fn extend(&self, buf: Vec<f32>) {
    match self.queue.write() {
      Ok(mut queue) => {
        queue.extend(buf);
      },
      Err(_) => {
        unreachable!("This shouldn't happen (extend)");
      },
    }
  }

  pub fn drain(&self) -> Vec<f32> {
    match self.queue.write() {
      Ok(mut queue) => {
        return queue.drain(..)
            .collect::<Vec<_>>();
      },
      Err(_) => {
        unreachable!("This shouldn't happen (drain)");
      },
    }
  }

  pub fn drain_size(&self, size: usize) -> Option<Vec<f32>> {
    match self.queue.write() {
      Ok(mut queue) => {
        if queue.len() < size {
          return None;
        }
        let vec = queue.drain(0..size)
            .collect::<Vec<_>>();

        return Some(vec);
      },
      Err(_) => {
        unreachable!("This shouldn't happen (drain_size)");
      },
    }
  }

  pub fn len(&self) -> usize {
    match self.queue.read() {
      Ok(mut queue) => {
        return queue.len();
      },
      Err(_) => {
        unreachable!("This shouldn't happen (len)");
      },
    }
  }
}
