use world_sys::core::synthesize::SynthesizeError;
use world_sys::core::synthesize::SynthesizeResult;
use world_sys::core::synthesize::synthesize;

use world_sys::core::spectral::code_spectral_envelope;

/*
TODO: ALGORITHM TO GET VOICE TRANSFORMATION:

- [mono]
- [librosa.resample (downsample)]
- core convert()
    - wav_padding()
        - [maths]
        - [np.pad]
    - world_decompose()
        - pyworld.harvest
        - pyworld.cheaptrick
        - pyworld.d4c
    - world_encode_spectral_envelope()
        - pyworld.code_spectral_envelope
    - [matrix transpose]
    - pitch_conversion()
        - [maths]
    - [maths : coded_sp_norm]
    - << evaluate ML model >>
    - [maths : coded_sp_converted]
    - [matrix transpose]
    - [np.ascontiguousarray]
    - world_decode_spectral_envelop
        - pyworld.get_cheaptrick_fft_size
        - pyworld.decode_spectral_envelope
    - world_speech_synthesis
        - pyworld.synthesize
    - [librosa.resample (upsample)]
    - [wav things]
*/

