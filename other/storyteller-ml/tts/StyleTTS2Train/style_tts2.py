import torchaudio
import librosa
from nltk.tokenize import word_tokenize

from models import *
from utils import *
from text_utils import TextCleaner
import torch
import random
import yaml

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

import nltk
nltk.download('punkt')

# load packages
import random
import yaml
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize

from models import *
from utils import *
from text_utils import TextCleaner

# load phonemizer
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True, words_mismatch='ignore')

from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from Utils.PLBERT.util import load_plbert
        
class StyleTTS2():
    def __init__(self,check_point:str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        self.mean, self.std = -4, 4
        self.text_cleaner = TextCleaner()
        
        self.config = yaml.safe_load(open("Models/LibriTTS/config.yml"))
        
        # load pretrained ASR model
        ASR_config = self.config.get('ASR_config', False)
        ASR_path = self.config.get('ASR_path', False)
        self.text_aligner = load_ASR_models(ASR_path, ASR_config)

        # load pretrained F0 model
        F0_path = self.config.get('F0_path', False)
        self.pitch_extractor = load_F0_models(F0_path)

        # load BERT model
        BERT_path = self.config.get('PLBERT_dir', False)
        self.plbert = load_plbert(BERT_path)
        
        self.model_params = recursive_munch(self.config['model_params'])
        self.model = build_model(self.model_params , self.text_aligner, self.pitch_extractor, self.plbert)
        _ = [self.model[key].eval() for key in self.model]
        _ = [self.model[key].to(self.device) for key in self.model]
            
        params_whole = torch.load(check_point, map_location=self.device)
        #params_whole = torch.load("Models/LibriTTS/epochs_2nd_00020.pth", map_location=self.device)
        # params_whole = torch.load("Models/LJSpeech/epoch_2nd_00100.pth", map_location='cpu')
        params = params_whole['net']

        for key in self.model:
            if key in params:
                print('%s loaded' % key)
                try:
                    self.model[key].load_state_dict(params[key])
                except:
                    from collections import OrderedDict
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    self.model[key].load_state_dict(new_state_dict, strict=False)
        #             except:
        #                 _load(params[key], self.model[key])
        _ = [self.model[key].eval() for key in self.model]
    
        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
            clamp=False
        )

    def length_to_mask(self,lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    def preprocess(self,wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor

    def compute_style(self,path):
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = self.preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)

    def inference(self,text, ref_s, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1):
        text = text.strip()
        ps = global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        tokens = self.text_cleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(self.device),
                                            embedding=bert_dur,
                                            embedding_scale=embedding_scale,
                                                features=ref_s, # reference from the same speaker as the embedding
                                                num_steps=diffusion_steps).squeeze(1)


            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
            s = beta * s + (1 - beta)  * ref_s[:, 128:]

            d = self.model.predictor.text_encoder(d_en,
                                            s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)


            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr,
                                    F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later

    def LFinference(self,text, s_prev, ref_s, alpha = 0.3, beta = 0.7, t = 0.7, diffusion_steps=5, embedding_scale=1):
        text = text.strip()
        ps = global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        ps = ps.replace('``', '"')
        ps = ps.replace("''", '"')

        tokens = self.text_cleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(self.device),
                                                embedding=bert_dur,
                                                embedding_scale=embedding_scale,
                                                features=ref_s, # reference from the same speaker as the embedding
                                                    num_steps=diffusion_steps).squeeze(1)

            if s_prev is not None:
                # convex combination of previous and current style
                s_pred = t * s_prev + (1 - t) * s_pred

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
            s = beta * s + (1 - beta)  * ref_s[:, 128:]

            s_pred = torch.cat([ref, s], dim=-1)

            d = self.model.predictor.text_encoder(d_en,
                                                s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)


            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr,F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        return out.squeeze().cpu().numpy()[..., :-100], s_pred # weird pulse at the end of the model, need to be fixed later

    
    def STinference(self,text, ref_s, ref_text, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1,speed = 1.0,pitch=1.0):
        text = text.strip()
        ps = global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)

        tokens = self.text_cleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        ref_text = ref_text.strip()
        ps = global_phonemizer.phonemize([ref_text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)

        ref_tokens = self.text_cleaner(ps)
        ref_tokens.insert(0, 0)
        ref_tokens = torch.LongTensor(ref_tokens).to(self.device).unsqueeze(0)


        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            ref_input_lengths = torch.LongTensor([ref_tokens.shape[-1]]).to(self.device)
            ref_text_mask = length_to_mask(ref_input_lengths).to(self.device)
            ref_bert_dur = self.model.bert(ref_tokens, attention_mask=(~ref_text_mask).int())
            s_pred = self.sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(self.device),
                                            embedding=bert_dur,
                                            embedding_scale=embedding_scale,
                                                features=ref_s, # reference from the same speaker as the embedding
                                                num_steps=diffusion_steps).squeeze(1)


            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
            s = beta * s + (1 - beta)  * ref_s[:, 128:]

            d = self.model.predictor.text_encoder(d_en,
                                            s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1) / speed
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)


            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
            F0_pred = F0_pred * pitch
            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr,
                                    F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later

from pathlib import Path
import time

def main():
    model = StyleTTS2(check_point="Models/LibriTTS/epochs_2nd_00020.pth")
    ref_text = "Didn't really have a body... I just kind of melted into shadows and creeped around on floors and walls, but now... I have this new job, and I just had to proove my self by bringing back a mortal soul."
    ref_s = Path("TestHarness/ds/698.wav")
    ref_s = model.compute_style(ref_s)
    
    text = "Didn't really have a body... I just kind of melted into shadows and creeped around on floors and walls, but now... I have this new job, and I just had to proove my self by bringing back a mortal soul."
    #output_wav = model.STinference(text, ref_s, ref_text, diffusion_steps=10, alpha=0.35, beta=0.7, embedding_scale=2.0)
    start_t = time.time()
    output_wav = model.STinference(text, ref_s, ref_text, diffusion_steps=100, alpha=0.15, beta=0.8, embedding_scale=1.5,speed=1.1,pitch=1.05)
    print(f'Took: {time.time()-start_t:.2f}s') 
    
    #output_wav = model.STinference(text, ref_s, ref_text, diffusion_steps=10, alpha=0.4, beta=0.7, embedding_scale=1.5)
    torchaudio.save("./test.wav", torch.tensor(output_wav).repeat(2,1), 24000)
    
def testbug():
    model = StyleTTS2(check_point="Models/LibriTTS/epochs_2nd_00020.pth")
    ref_s = Path("TestHarnessV2/goku/381.wav")
    ref_s = model.compute_style(ref_s)
    
    ref_text = "I thought that would do more damage than that. Has awakened the legendary warrior that lived in my heart! I am Goku and I am a Super Saiyan! So your finally going to fight with hundred percent of your strength."
    # if embedding scale is too high it could produce a lot of noise.
    output_wav = model.STinference("This is.", ref_s, ref_text, diffusion_steps=10, alpha=0.45, beta=0.6, embedding_scale=1.0)
    torchaudio.save("./test.wav", torch.tensor(output_wav).repeat(2,1), 24000)
if __name__ == "__main__":
    #main()
    testbug()