//! Send "Unreal LiveLink" network packets that were recorded from Xsens Animate
//! This should help bypass XSens ridiculous software.

use anyhow::Result as AnyhowResult;
use clap::{App, Arg};
use log::info;
use pcap_parser::*;
use pcap_parser::traits::PcapReaderIterator;
use std::fs::File;
use std::time::Duration;
use tokio::net::UdpSocket;

const LOCAL_ADDR : &'static str = "0.0.0.0:8022";

//const REMOTE_ADDR : &'static str = "192.168.50.31:9763";
const REMOTE_ADDR : &'static str = "127.0.0.1:9763";

const DEFAULT_ANIMATION_PACKET_FILE : &'static str = "three_dancers_animation.pcap";

#[tokio::main]
pub async fn main() -> AnyhowResult<()> {
  let matches = App::new("xsens-packet-send")
      .arg(Arg::with_name("file")
          .short("f")
          .long("file")
          .value_name("FILE")
          .help("Packet file to read")
          .takes_value(true))
      .arg(Arg::with_name("host")
          .short("h")
          .long("host")
          .value_name("HOST")
          .help("Host and port to send traffic to")
          .takes_value(true))
      .get_matches();

  let animation_file = matches.value_of("file")
      .unwrap_or(DEFAULT_ANIMATION_PACKET_FILE);

  let hostname = matches.value_of("host")
      .unwrap_or(REMOTE_ADDR);

  info!("Connecting to Unreal animation host: {}", hostname);

  let socket = UdpSocket::bind(LOCAL_ADDR).await?;
  socket.connect(hostname).await?;

  loop {
    play_legacy_pcap_file_format(animation_file, &socket).await?;
    tokio::time::sleep(Duration::from_millis(500)).await;
  }

  Ok(())
}

pub async fn play_legacy_pcap_file_format(animation_file: &str, socket: &UdpSocket) -> AnyhowResult<()> {
  info!("Reading config file: {}", animation_file);

  let file = File::open(animation_file)?;

  let mut num_blocks = 0;
  let mut reader = LegacyPcapReader::new(65536, file)?;

  loop {
    match reader.next() {
      Ok((offset, block)) => {
        num_blocks += 1;

        match block {
          PcapBlockOwned::LegacyHeader(header) => {
          },
          PcapBlockOwned::Legacy(block) => {
            // NB(brandon): The first 32 bytes seem to be the UDP packet metadata.
            let slice = &block.data[32..];
            let len = socket.send(slice).await?;
            //println!("Sent: {}", len);

            tokio::time::sleep(Duration::from_millis(5)).await;
          },
          PcapBlockOwned::NG(_) => unreachable!(),
        }
        reader.consume(offset);
      },
      Err(PcapError::Eof) => break,
      Err(PcapError::Incomplete) => {
        reader.refill().unwrap();
      },
      Err(e) => panic!("error while reading: {:?}", e),
    }
  }

  println!("num_blocks: {}", num_blocks);
  Ok(())
}
