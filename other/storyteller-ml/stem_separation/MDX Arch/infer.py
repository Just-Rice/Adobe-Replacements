import numpy as np
import torch
import torchaudio
import onnxruntime as ort
import librosa

class MDXModel:

    def __init__(self, chunk_size, n_fft, hop, dim_f, dim_t, dim_c, model_file):
        self.chunk_size = chunk_size
        self.n_fft = n_fft
        self.hop = hop
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.dim_c = dim_c
        self.n_bins = self.n_fft//2+1
        self.trim = self.n_fft//2
        self.window = torch.hann_window(self.n_fft)
        self.freq_pad = torch.zeros([1, self.dim_c, self.n_bins-self.dim_f, self.dim_t])
        self.adjust = 1
        self.ort_ = ort.InferenceSession(
            "models/" + model_file,
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider"
            ]
        )
        self.model_run = lambda spek:self.ort_.run(None, {'input': spek})[0]


    def stft(self, x):
            x = x.reshape([-1, self.chunk_size])
            x = torch.stft(torch.from_numpy(x), n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True,return_complex=True)
            x=torch.view_as_real(x)
            x = x.permute([0,3,1,2])
            x = x.reshape([-1,self.dim_c,self.n_bins,self.dim_t])
            return x[:,:,:self.dim_f]

    def istft(self, x, freq_pad=None):
            freq_pad = self.freq_pad.repeat([x.shape[0],1,1,1]) if freq_pad is None else freq_pad
            x = torch.cat([x, freq_pad], -2)
            x = x.reshape([-1,2,2,self.n_bins,self.dim_t]).reshape([-1,2,self.n_bins,self.dim_t])
            x = x.permute([0,2,3,1])
            x=x.contiguous()
            x=torch.view_as_complex(x)
            x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
            return x.reshape([-1,2,self.chunk_size])

    def run_model(self, mix, is_ckpt=False, is_match_mix=False):
            spek = self.stft(mix)*self.adjust
            spek[:, :, :3, :] *= 0
            spec_pred = self.model_run(spek.float().numpy())
            stem1 = self.istft(torch.from_numpy(spec_pred)).transpose(0,1).reshape(2, -1).numpy()
            return stem1

    def process(self, input_file_path, stem1_output_path, stem2_output_path):
        wav, _ = librosa.load(input_file_path, sr=44100, mono=False)
        if wav.ndim == 1:
            wav = np.array([wav, wav])
        if wav.ndim != 2:
            raise Exception("Only mono or stereo files are supported")
        full_size = ((wav.shape[1] // self.chunk_size)+1)*self.chunk_size
        full_stem1 = np.ndarray([2, full_size])
        full_stem2 = np.ndarray([2, full_size])
        for i in range(0, wav.shape[1], self.chunk_size):
            chunk_wav_pad = np.zeros([2, self.chunk_size])
            if i + self.chunk_size < wav.shape[1]:
                chunk_wav_pad[:2, :self.chunk_size] = wav[:2, i:i+self.chunk_size]
            else:
                chunk_wav_pad[:2, i+self.chunk_size:full_size] = wav[:2, i*self.chunk_size:full_size]
            chunk_stem1 = self.run_model(chunk_wav_pad)
            chunk_wav_pad, chunk_stem1 = normalize_two_stem(chunk_wav_pad, chunk_stem1)
            chunk_stem2 = chunk_wav_pad - chunk_stem1
            full_stem1[:, i:i+self.chunk_size] = chunk_stem1
            full_stem2[:, i:i+self.chunk_size] = chunk_stem2
        # trim padding
        full_stem1 = full_stem1[:2, :wav.shape[1]]
        full_stem2 = full_stem2[:2, :wav.shape[1]]

        torchaudio.save(stem1_output_path, torch.from_numpy(full_stem1).float(), 44100)
        torchaudio.save(stem2_output_path, torch.from_numpy(full_stem2).float(), 44100)

# Not totally sure if this is doing anything useful
def normalize_two_stem(wave, mix, is_normalize=False):
    maxv = np.abs(wave).max()
    max_mix = np.abs(mix).max()
    if maxv > 1.0:
        if is_normalize:
            wave /= maxv
            mix /= maxv
    return wave, mix


class MDXNetMain(MDXModel):
    def __init__(self):
        super(MDXNetMain, self).__init__(261120, 7680, 1024, 3072, 256, 4, "UVR_MDXNET_Main.onnx")

class MDXNetKara2(MDXModel):
    def __init__(self):
        super(MDXNetKara2, self).__init__(261120, 7680, 1024, 2048, 256, 4, "UVR_MDXNET_KARA_2.onnx")

# NB (Paul): This model kind of sucked when I tested it with eye of the tiger
class ReverbHQ(MDXModel):
    def __init__(self):
        super(ReverbHQ, self).__init__(523264, 6144, 1024, 3072, 512, 4, "Reverb_HQ_By_FoxJoy.onnx")


if __name__ == "__main__":
    #ReverbHQ().process("/mnt/8terra/tts/voice-conversion-sources/eyeofthetiger.wav", "/tmp/stem1.wav", "/tmp/stem2.wav")
    MDXNetMain().process("/tmp/Carly Rae Jepsen - Call Me Maybe [fWNaR-rxAic].webm", "stem1.wav", "stem2.wav")
