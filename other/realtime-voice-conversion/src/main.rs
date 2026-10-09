//!
//! This is a process that captures microphone input, ships proto-over-zeromq to a python
//! CycleGAN-VC (tensorflow) sidecar server, collects the results, then plays them over
//! the speaker.
//!

extern crate bytes;
extern crate byteorder;
extern crate cpal;
extern crate failure;
extern crate prost;
extern crate prost_types;
extern crate wavy;
extern crate world_sys;
extern crate zmq;

//include!(concat!(env!("OUT_DIR"), "/voder.audio.rs"));

pub mod audio_hardware;
pub mod audio_queue;
//pub mod model;
pub mod protos;
pub mod sidecar;
pub mod synthesis;

use audio_hardware::AudioHardware;
use protos::voder_audio::VocodeAudioRequest;
use protos::voder_audio::VocodeAudioResponse;
use sidecar::Sidecar;

use wavy::*;

use byteorder::{ByteOrder, BigEndian, LittleEndian, ReadBytesExt};
use failure::_core::time::Duration;
use audio_queue::AudioQueue;
use prost::Message;
//use model::load_model;
//use model::print_version;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{Read, Cursor};
use std::path::Path;
use std::process::exit;
use std::sync::Arc;
use std::thread;
use zmq::{Error, Socket};
use protos::voder_audio::vocode_audio_request::VocodeParams;

fn main() {
  //print_version();
  //load_model(); // TODO: This works. Temporarily commented out
  run_audio().expect("Should work");
}

const LATENCY_MS: f32 = 50.0;

fn run_audio() -> Result<(), failure::Error> {
  let mut microphone_queue = Arc::new(AudioQueue::new());
  let mut microphone_queue_2 = microphone_queue.clone();

  let mut post_process_queue = Arc::new(AudioQueue::new());
  let mut post_process_queue_2 = post_process_queue.clone();
  let mut sidecar = Sidecar::new("tcp://127.0.0.1:5555", microphone_queue_2, post_process_queue);

  thread::spawn(move || {
    sidecar.run();
  });

  let mut audio_hardware = AudioHardware::new(microphone_queue, post_process_queue_2).expect("");
  audio_hardware.run();
  Ok(())
}
