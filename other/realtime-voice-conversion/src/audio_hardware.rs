use cpal::traits::{DeviceTrait, EventLoopTrait, HostTrait};
use cpal;
use audio_queue::AudioQueue;
use std::sync::Arc;

/// CPAL Audio Device
/// Access audio input and output streams.
pub struct AudioHardware {
  host: cpal::Host,
  event_loop: cpal::EventLoop,
  microphone_queue: Arc<AudioQueue>,
  speaker_queue: Arc<AudioQueue>,
}

impl AudioHardware {
  pub fn new(microphone_queue: Arc<AudioQueue>, speaker_queue: Arc<AudioQueue>) -> Result<Self, failure::Error> {
    let host = cpal::default_host();
    let event_loop = host.event_loop();

    let input_device = host.default_input_device().expect("failed to get default input device");
    let output_device = host.default_output_device().expect("failed to get default output device");

    println!("Using default input device: \"{}\"", input_device.name()?);
    println!("Using default output device: \"{}\"", output_device.name()?);

    // We'll try and use the same format between streams to keep it simple
    let mut format = input_device.default_input_format()?;
    format.data_type = cpal::SampleFormat::F32;

    // Build streams.
    println!("Attempting to build both streams with `{:?}`.", format);
    let input_stream_id = event_loop.build_input_stream(&input_device, &format)?;
    let output_stream_id = event_loop.build_output_stream(&output_device, &format)?;
    println!("Successfully built streams.");

    Ok(AudioHardware {
      host,
      event_loop,
      microphone_queue,
      speaker_queue,
    })
  }

  pub fn run(&mut self) {
    self.event_loop.run(|id, result| {
      let data = match result {
        Ok(data) => data,
        Err(err) => {
          eprintln!("An error occurred on stream {:?}: {}", id, err);
          return;
        }
      };

      match data {
        cpal::StreamData::Input { buffer: cpal::UnknownTypeInputBuffer::F32(buffer) } => {
          //assert_eq!(id, input_stream_id);
          let mut output_fell_behind = false;
          for &sample in buffer.iter() {
            self.microphone_queue.push_back(sample);
          }
        },
        cpal::StreamData::Output { buffer: cpal::UnknownTypeOutputBuffer::F32(mut buffer) } => {
          //println!("Audio out buffer len: {}", self.speaker_queue.len());
          let request_size = buffer.len();
          let mut drained = self.speaker_queue.drain_size((request_size));
          match drained {
            None => {
              for sample in buffer.iter_mut() {
                *sample = 0.0;
              }
            },
            Some(mut drained) => {
              for (i, sample) in buffer.iter_mut().enumerate() {
                *sample = drained.get(i).copied().unwrap();
                //*sample = 0.0;
              }
            },
          }
        },
        _ => panic!("We're expecting f32 data."),
      }
    })
  }
}
