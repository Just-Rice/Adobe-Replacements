use audio_queue::AudioQueue;
use prost::Message;
use protos::voder_audio::VocodeAudioRequest;
use protos::voder_audio::VocodeAudioResponse;
use protos::voder_audio::vocode_audio_request::VocodeParams;
use std::collections::VecDeque;
use std::sync::Arc;
use zmq::{Error, Socket};
use zmq;

/// Talks a custom protobuf over ZeroMQ protocol to a Python TensorFlow sidecar process
/// that evaluates the model and ships back audio byte results.
pub struct Sidecar {
  socket_endpoint: String,
  microphone_queue: Arc<AudioQueue>,
  processed_queue: Arc<AudioQueue>,
}

impl Sidecar {
  pub fn new(socket_endpoint: &str,
             microphone_queue: Arc<AudioQueue>,
             processed_queue: Arc<AudioQueue>) -> Self {
    Self {
      socket_endpoint: socket_endpoint.into(),
      microphone_queue,
      processed_queue,
    }
  }

  /// Heuristic to determine if a speaker is speaking.
  /// Basically just checks audio loudness.
  fn is_speaking(audio: &Vec<f32>) -> bool {
    let mut avg_pos = 0.0f32;
    let mut num_pos = 0;

    for x in audio {
      if x > &0.0f32 {
        avg_pos += x;
        num_pos += 1;
      }
    }

    avg_pos = avg_pos / num_pos as f32;
    avg_pos > 0.006f32
  }

  pub fn run(&mut self) {
    let mut context = zmq::Context::new();
    let mut socket = context.socket(zmq::REQ).unwrap();

    socket.connect(&self.socket_endpoint).unwrap();

    let mut reconnect = false;
    let mut fail_count = 0;

    const SEND_SIZE : usize = 5000;

    let mut request_batch_number = 0i64;

    let mut activated = false;
    let mut frames_activated = 0;
    let mut frames_deactivated = 0;

    let mut ring_buffer = VecDeque::with_capacity(5);

    loop {
      let mut drained = match self.microphone_queue.drain_size(SEND_SIZE) {
        None => { continue; },
        Some(d) => d,
      };

      /*
      Heuristic for audio. Don't record when silent (using threshold).

        mic: act act act dec act act act act dec dec dec dec dec act dec
        sent:         | ok ok ok ok ok ok ok ok ok ok ok|
      */


      // NB: We don't want to lose audio right at the activating edge.
      // Save some bounded history to replay once activated.
      ring_buffer.push_back(drained.clone());
      if ring_buffer.len() > 5 {
        ring_buffer.pop_front();
      }

      if activated {
        if !Self::is_speaking(&drained) {
          frames_deactivated += 1;
        } else {
          frames_deactivated = 0;
        }

        if frames_deactivated > 20 {
          println!("Deactivating edge");
          activated = false;
          frames_activated = 0;
          frames_deactivated = 0;
        }

      } else {
        if Self::is_speaking(&drained) {
          frames_activated += 1;
        } else {
          frames_activated = 0;
        }

        if frames_activated > 3 {
          println!("Activating edge");
          activated = true;
          frames_activated = 0;
          frames_deactivated = 0;
        }
      }

      if !activated {
        continue;
      }

      let drained = ring_buffer.drain(..)
          .flat_map(|sample| sample)
          .collect::<Vec<_>>();

      request_batch_number += 1;

      let mut vocode_request = VocodeAudioRequest::default();
      //vocode_request.sample_rate = 16000;
      //vocode_request.output_rate = 16000;
      //vocode_request.model_sampling_rate = 88000; // This is close!
      //vocode_request.model_sampling_rate = 88000;
      //vocode_request.skip_resample = false;
      vocode_request.skip_vocode = false;
      vocode_request.discard_vocoded_audio = false;
      //vocode_request.buffer_size_minimum = 5000; // AWFUL. SO CHOPPY.
      //vocode_request.buffer_size_minimum = 50000; // Practically real time, but lots more phase distortion.
      //vocode_request.buffer_size_minimum = 30000; // Hm, hmm... not bad
      vocode_request.buffer_size_minimum = 70000; // Sounds pretty good.
      //vocode_request.buffer_size_minimum = 100000; // This sounds good! A bit slow ~3seconds.
      //vocode_request.buffer_size_minimum = 200000;
      vocode_request.float_audio = drained.clone();
      vocode_request.request_batch_number = request_batch_number;

      let mut vocode_params = VocodeParams::default();
      vocode_params.original_source_rate = 88000; // This is correct for saving

      vocode_params.pre_convert_resample = true;
      vocode_params.pre_convert_resample_rate = 16000;

      vocode_params.model_hyperparameter_sampling_rate = 16000;

      vocode_params.post_convert_resample = true;
      vocode_params.post_convert_resample_rate = 88000;

      vocode_params.original_source_save_file = false;
      vocode_params.pre_convert_resample_save_file = false;
      vocode_params.model_save_file = false;
      vocode_params.post_convert_resample_save_file = false;

      vocode_request.vocode_params = Some(vocode_params);

      let mut encoded_bytes = Vec::with_capacity(vocode_request.encoded_len());
      vocode_request.encode(&mut encoded_bytes).unwrap();

      if reconnect {
        reconnect = false;
        println!("RECONNECT");

        //thread::sleep(Duration::from_millis(200));

        socket = match context.socket(zmq::REQ) {
          Ok(s) => s,
          Err(e) => {
            println!("Error creating socket: {:?}", e);
            continue
          },
        };

        match socket.connect("tcp://127.0.0.1:5555") {
          Ok(_) => {},
          Err(err) => {
            println!("Err B: {:?}", err);
          },
        }
      }

      match socket.send(&encoded_bytes, 0) {
        Ok(_) => {
          //println!("Sent len: {}", bytes.len());
        },
        Err(e) => {
          println!("send err {}: {:?}", e.to_raw(), e);
          fail_count += 1;
        },
      }

      match socket.recv_bytes(0) {
        Ok(buf) => {
          //println!("---> Buf len: {}", buf.len());
          if buf.len() > 2 {
            // Receive data condition.
            let vocode_response = VocodeAudioResponse::decode(buf).unwrap();
            self.processed_queue.extend(vocode_response.float_audio);
          }
        },
        Err(e) => {
          fail_count += 1;
          println!("recv err: {:?}", e);
        },
      }

      if fail_count > 5 {
        fail_count = 0;
        reconnect = true;
      }
    }
  }
}