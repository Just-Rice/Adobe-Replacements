extern crate prost_build;

fn main() {
  //prost_build::compile_protos(&["protos/audio.proto"],
  //                        &["protos/"]).unwrap();
  prost_build::Config::new()
      .out_dir("src/protos/")
      .compile_protos(&["protos/audio.proto"],
                      &["protos/"])
      .unwrap();
}