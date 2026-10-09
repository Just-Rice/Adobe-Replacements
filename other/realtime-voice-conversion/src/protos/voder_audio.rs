#[derive(Clone, PartialEq, ::prost::Message)]
pub struct VocodeAudioRequest {
    #[prost(float, repeated, tag="1")]
    pub float_audio: ::std::vec::Vec<f32>,
    /// An autoincrement sent with each request.
    #[prost(int64, tag="2")]
    pub request_batch_number: i64,
    #[prost(message, optional, tag="10")]
    pub vocode_params: ::std::option::Option<vocode_audio_request::VocodeParams>,
    // We'll resend these params for every batch, but will only use
    // the last batch's settings. Kind of a dumb API, but it'll work.
    //int32 sample_rate = 2;

    ///bool skip_resample = 3;
    #[prost(bool, tag="4")]
    pub skip_vocode: bool,
    // Debugging
    //bool save_files = 5;

    /// How big we let the buffer grow before running 'convert'.
    #[prost(int32, tag="6")]
    pub buffer_size_minimum: i32,
    // Requested output rate
    //int32 output_rate = 7;

    /// Discard the vocoded audio and return the original
    #[prost(bool, tag="8")]
    pub discard_vocoded_audio: bool,
}
pub mod vocode_audio_request {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct VocodeParams {
        /// The initial sample rate coming from the microphone.
        #[prost(int32, tag="1")]
        pub initial_sample_rate: i32,
        /// Original sample
        #[prost(int32, tag="9")]
        pub original_source_rate: i32,
        #[prost(bool, tag="10")]
        pub original_source_save_file: bool,
        /// Resample before passing to the algorithm?
        #[prost(bool, tag="2")]
        pub pre_convert_resample: bool,
        #[prost(int32, tag="3")]
        pub pre_convert_resample_rate: i32,
        #[prost(bool, tag="4")]
        pub pre_convert_resample_save_file: bool,
        /// The hard-coded hyperparameter of the model.
        /// Probably don't want to change from 16000.
        #[prost(int32, tag="5")]
        pub model_hyperparameter_sampling_rate: i32,
        #[prost(bool, tag="11")]
        pub model_save_file: bool,
        /// Resample after passing to the algorithm?
        #[prost(bool, tag="6")]
        pub post_convert_resample: bool,
        #[prost(int32, tag="7")]
        pub post_convert_resample_rate: i32,
        #[prost(bool, tag="8")]
        pub post_convert_resample_save_file: bool,
    }
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct VocodeAudioResponse {
    #[prost(float, repeated, tag="1")]
    pub float_audio: ::std::vec::Vec<f32>,
    /// A sidecar-side autoincrement sent with each request.
    #[prost(int64, tag="2")]
    pub response_batch_number: i64,
}
