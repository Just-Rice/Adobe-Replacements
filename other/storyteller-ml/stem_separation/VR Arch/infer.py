# stdlib
import math
import os

# pypi
import torch, torchaudio
import librosa
import numpy as np

# local
from model_param_init import ModelParameters
import nets_new
import spec_utils

class VR_Arch:
    def __init__(self, model_path, model_param_json):
        self.mp = ModelParameters(model_param_json)
        self.model_path = model_path
        self.model_capacity = 32, 128
        self.high_end_process = 'none'
        self.window_size = 512
        self.is_tta = False
        self.batch_size = 1
        self.aggressiveness = {
            'value': 10,
            'split_bin': self.mp.param['band'][1]['crop_stop'],
            'aggr_correction': self.mp.param.get('aggr_correction')
        }
        self.is_post_process = True
        self.is_normalization = True
        self.post_process_threshold = 0.2
        self.model_samplerate = 44100


    def seperate(self, source_path, primary_stem_path, secondary_stem_path):
        self.audio_file = source_path
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        nn_arch_sizes = [
            31191, # default
            33966, 56817, 123821, 123812, 129605, 218409, 537238, 537227]
        vr_5_1_models = [56817, 218409]
        model_size = math.ceil(os.stat(self.model_path).st_size / 1024)
        nn_arch_size = min(nn_arch_sizes, key=lambda x:abs(x-model_size))

        if nn_arch_size in vr_5_1_models or self.is_vr_51_model:
            self.model_run = nets_new.CascadedNet(self.mp.param['bins'] * 2, nn_arch_size, nout=self.model_capacity[0], nout_lstm=self.model_capacity[1])
        else:
            self.model_run = nets.determine_model_capacity(self.mp.param['bins'] * 2, nn_arch_size)
        self.model_run.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.model_run.to(device)

        y_spec, v_spec = self.inference_vr(self.loading_mix(), device, self.aggressiveness)


        self.primary_source = spec_utils.normalize(self.spec_to_wav(y_spec), self.is_normalization).T
        if not self.model_samplerate == 44100:
            self.primary_source = librosa.resample(self.primary_source.T, orig_sr=self.model_samplerate, target_sr=44100)
        torchaudio.save(primary_stem_path, torch.from_numpy(self.primary_source).float().T, 44100)

        self.secondary_source = self.spec_to_wav(v_spec)
        self.secondary_source = spec_utils.normalize(self.spec_to_wav(v_spec), self.is_normalization).T
        if not self.model_samplerate == 44100:
            self.secondary_source = librosa.resample(self.secondary_source.T, orig_sr=self.model_samplerate, target_sr=44100)
        torchaudio.save(secondary_stem_path, torch.from_numpy(self.secondary_source).float().T, 44100)
        torch.cuda.empty_cache()

    def loading_mix(self):
        X_wave, X_spec_s = {}, {}
        bands_n = len(self.mp.param['band'])
        for d in range(bands_n, 0, -1):
            bp = self.mp.param['band'][d]
            wav_resolution = bp['res_type']
            if d == bands_n: # high-end band
                X_wave[d], _ = librosa.load(self.audio_file, bp['sr'], False, dtype=np.float32, res_type=wav_resolution)
                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
            else: # lower bands
                X_wave[d] = librosa.resample(X_wave[d+1], self.mp.param['band'][d+1]['sr'], bp['sr'], res_type=wav_resolution)
            X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(X_wave[d], bp['hl'], bp['n_fft'], self.mp.param['mid_side'],
                                                            self.mp.param['mid_side_b2'], self.mp.param['reverse'])
            if d == bands_n and self.high_end_process != 'none':
                self.input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (self.mp.param['pre_filter_stop'] - self.mp.param['pre_filter_start'])
                self.input_high_end = X_spec_s[d][:, bp['n_fft']//2-self.input_high_end_h:bp['n_fft']//2, :]

        X_spec = spec_utils.combine_spectrograms(X_spec_s, self.mp)
        del X_wave, X_spec_s

        return X_spec

    def inference_vr(self, X_spec, device, aggressiveness):
        def _execute(X_mag_pad, roi_size):
            X_dataset = []
            patches = (X_mag_pad.shape[2] - 2 * self.model_run.offset) // roi_size
            total_iterations = patches//self.batch_size if not self.is_tta else (patches//self.batch_size)*2
            for i in range(patches):
                start = i * roi_size
                X_mag_window = X_mag_pad[:, :, start:start + self.window_size]
                X_dataset.append(X_mag_window)

            X_dataset = np.asarray(X_dataset)
            self.model_run.eval()
            with torch.no_grad():
                mask = []
                for i in range(0, patches, self.batch_size):
                    X_batch = X_dataset[i: i + self.batch_size]
                    X_batch = torch.from_numpy(X_batch).to(device)
                    pred = self.model_run.predict_mask(X_batch)
                    if not pred.size()[3] > 0:
                        raise Exception(ERROR_MAPPER[WINDOW_SIZE_ERROR])
                    pred = pred.detach().cpu().numpy()
                    pred = np.concatenate(pred, axis=2)
                    mask.append(pred)
                if len(mask) == 0:
                    raise Exception(ERROR_MAPPER[WINDOW_SIZE_ERROR])
                mask = np.concatenate(mask, axis=2)
            return mask

        def postprocess(mask, X_mag, X_phase):
            is_non_accom_stem = False
            #for stem in NON_ACCOM_STEMS:
            #    if stem == self.primary_stem:
            #        is_non_accom_stem = True
            mask = spec_utils.adjust_aggr(mask, is_non_accom_stem, aggressiveness)

            if self.is_post_process:
                mask = spec_utils.merge_artifacts(mask, thres=self.post_process_threshold)

            y_spec = mask * X_mag * np.exp(1.j * X_phase)
            v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)
            return y_spec, v_spec

        X_mag, X_phase = spec_utils.preprocess(X_spec)
        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = spec_utils.make_padding(n_frame, self.window_size, self.model_run.offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()
        mask = _execute(X_mag_pad, roi_size)

        if self.is_tta:
            pad_l += roi_size // 2
            pad_r += roi_size // 2
            X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
            X_mag_pad /= X_mag_pad.max()
            mask_tta = _execute(X_mag_pad, roi_size)
            mask_tta = mask_tta[:, :, roi_size // 2:]
            mask = (mask[:, :, :n_frame] + mask_tta[:, :, :n_frame]) * 0.5
        else:
            mask = mask[:, :, :n_frame]

        y_spec, v_spec = postprocess(mask, X_mag, X_phase)
        return y_spec, v_spec

    def spec_to_wav(self, spec):
        if self.high_end_process.startswith('mirroring'): 
            input_high_end_ = spec_utils.mirroring(self.high_end_process, spec, self.input_high_end, self.mp)
            wav = spec_utils.cmb_spectrogram_to_wave(spec, self.mp, self.input_high_end_h, input_high_end_)
        else:
            wav = spec_utils.cmb_spectrogram_to_wave(spec, self.mp)
        return wav

class UVR_DeEcho_DeReverb(VR_Arch):
    def __init__(self):
        super(UVR_DeEcho_DeReverb, self).__init__("models/UVR-DeEcho-DeReverb.pth", "models/data/4band_v3.json")


if __name__ == "__main__":
    UVR_DeEcho_DeReverb().seperate(
        "/mnt/8terra/tts/voice-conversion-sources/eyeofthetiger.wav",
        "dereverb.wav",
        "reverb.wav"
    )
