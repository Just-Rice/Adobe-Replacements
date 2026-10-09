//! Generate random points to send over ZeroMQ to a consumer

mod point;
mod color;

use point::Point;
use std::thread;
use zmq::{Error, Socket, Context, DONTWAIT};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use crate::color::Color;

const SOCKET_ADDRESS : &'static str = "tcp://127.0.0.1:8888";

enum MessagingState {
  Sending,
  Receiving,
}

fn main() {
  let context = zmq::Context::new();

  //let socket = ctx.socket(zmq::PUSH).unwrap();
  let mut socket = context.socket(zmq::REQ).unwrap();

  //socket.connect("tcp://127.0.0.1:5555").unwrap();
  //socket.bind("tcp://127.0.0.1:5555").unwrap();
  socket.connect(SOCKET_ADDRESS).unwrap();

  let mut reconnect =  false;
  let mut fail_count = 0;
  let mut messaging_state = MessagingState::Sending;
  let mut color = Color::Blue;

  loop {
    let color = get_color();
    let point = Point::at_random_range(-1000.0f32, 1000.0f32, color);
    let bytes = point.to_bytes();
    //println!("Point : {}", point.debug_string());

    if reconnect {
      socket = reconnect_socket(&context, socket, SOCKET_ADDRESS);
      reconnect = false;
      messaging_state = MessagingState::Sending;
    }

    match messaging_state {
      MessagingState::Sending => {
        //println!("Sending request...");
        //let result = socket.send("hello world!", 0);
        let result = socket.send(&bytes, 0);

        match result {
          Ok(_) => {
            //println!("Sent!");
            //thread::sleep(Duration::from_millis(250));
            messaging_state = MessagingState::Receiving;
          },
          Err(e) => {
            //eprintln!("Send Error ({}): {:?}", e.to_raw(), e);
            //thread::sleep(Duration::from_millis(250));
            fail_count += 1;
          },
        }

      },
      MessagingState::Receiving => {
        //println!("Awaiting response...");
        let result = socket.recv_bytes(DONTWAIT);

        match result {
          Ok(_) => {
            //println!("Response received!");
            messaging_state = MessagingState::Sending;
          },
          Err(e) => {
            //eprintln!("Recv Error ({}): {:?}", e.to_raw(), e);
            //thread::sleep(Duration::from_millis(250));
            fail_count += 1;
          },
        }

      },
    }

    if fail_count > 5 {
      reconnect = true;
      fail_count = 0;
      //thread::sleep(Duration::from_millis(2000));
    }
  }
}

fn reconnect_socket(context: &Context, socket: Socket, address: &str) -> Socket {
  //println!("[reconnect] Creating new socket...");

  let mut socket = match context.socket(zmq::REQ) {
    Ok(s) => {
      //println!("New socket created.");
      s
    },
    Err(e) => {
      //println!("Error creating new socket: {:?}", e);
      return socket;
    },
  };

  //println!("Connecting new socket...");
  match socket.connect(address) {
    Ok(_) => {
      //println!("New socket connected.");
    },
    Err(err) => {
      //println!("Error connecting new socket: {:?}", err);
    },
  }

  return socket;
}

fn get_color() -> Color {
  let start = SystemTime::now();
  let timestamp = start
      .duration_since(UNIX_EPOCH)
      .expect("Time should work.");

  let seconds = timestamp.as_secs();

  match seconds % 10 {
    0 | 1 => Color::Black,
    2 | 3 => Color::Blue,
    4 | 5 => Color::Red,
    6 | 7 => Color::Green,
    8 | 9 => Color::White,
    _ => Color::White,
  }
}