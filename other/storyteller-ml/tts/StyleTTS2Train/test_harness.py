import wandb
from pathlib import Path
from collections import defaultdict
from style_tts2 import StyleTTS2
import torch
import torchaudio

class CharacterData:
    def __init__(self,full_path,char_name,file_name,ref_text) -> None:
        self.char_name = char_name
        self.file_name = file_name
        self.ref_text = ref_text
        self.full_path = full_path
        
class TestHarness:
    def __init__(self):
        # # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="Style TTS 2 Finetune",
            # track hyperparameters and run metadata
            config={
                "architecture": "Style TTS 2",
            }
        )
        
        self.test_text = ["This is!","Hello world testing!""This is a slightly... longer test.""This is a longer test, of style TTS Two!", "This should be a lengthy sentence."]
        self.test_characters = self.get_characters()

    def log_values(self,loss_gen_all, d_loss, loss_ce, loss_dur, loss_lm, loss_norm_rec, 
                   loss_F0_rec, loss_sty, loss_diff, d_loss_slm, loss_gen_lm, iters, epoch, step, 
                   log_interval, running_loss,train_list_len,batch_size):
        wandb.log({
                'train/mel_loss': running_loss / log_interval,
                'train/gen_loss': loss_gen_all,
                'train/d_loss': d_loss,
                'train/ce_loss': loss_ce,
                'train/dur_loss': loss_dur,
                'train/slm_loss': loss_lm,
                'train/norm_loss': loss_norm_rec,
                'train/F0_loss': loss_F0_rec,
                'train/sty_loss': loss_sty,
                'train/diff_loss': loss_diff,
                'train/d_loss_slm': d_loss_slm,
                'train/gen_loss_slm': loss_gen_lm,
                'train/iters':iters,
                'epoch':epoch+1,
                'step':step,
                'total_steps':train_list_len//batch_size
            })
        
    def log_eval(self,mel_loss,dur_loss,F0_loss,epoch):
        wandb.log({
            'eval/mel_loss':mel_loss,
            'eval/dur_loss':dur_loss,
            'eval/F0_loss':F0_loss,
            'epoch':epoch+1
        })
        
    def sample_ood_voice(self,file:str="testing.wav",sample_rate=24000,character:str="character_name",caption:str="What is supposed to be said"):
        wandb.log({character: wandb.Audio(file, sample_rate=sample_rate, caption=caption)})

    def _next_check_point_inference(self,file_input,text,checkpoint_name:str,callback):
        saved_path = callback(file_input,text,checkpoint_name)
        return saved_path
    
    def test(self,check_point:Path,top=1):
        model = StyleTTS2(check_point)

        char_name_ls = TestHarness.get_characters()
        count = 0
        for char_name in char_name_ls.keys():
            char_sample = char_name_ls[char_name]
            for sample in char_sample:
                s_ref = Path(sample.full_path)
                s_ref = model.compute_style(s_ref)
                for i,text in enumerate(self.test_text):
                    mono_wav = model.STinference(text=text,
                                        ref_text=sample.ref_text,
                                        ref_s=s_ref,
                                        diffusion_steps=10,alpha=0.45, beta=0.6, embedding_scale=1.0)
                    stereo_wav = torch.tensor(mono_wav).repeat(2,1)
                    path = f"./test/{char_name}|{i}-text|{sample.file_name}"
                    torchaudio.save(path,stereo_wav,sample_rate=24000)
                    self.sample_ood_voice(file=path,sample_rate=24000,character=char_name,caption=f"Text: {text} | Reference Text:{sample.ref_text}")
                    
            count += 1
            if count == top:
                break
    @staticmethod
    def get_lines_from_file(file_path):
        with open(file_path,'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            return lines           
    
    @staticmethod
    def get_characters():
        characters = dict()
        lines = TestHarness.get_lines_from_file("TestHarnessV2/info.txt")
        
        for line in lines:
            comp = line.split("|")
            full_path = comp[0]
            pieces = full_path.split("/")
            char_name = pieces[1]
            file_name = pieces[2]
            ref_text = comp[1]
            
            base_path = "TestHarnessV2"
            
            if char_name in characters:
                characters[char_name].append(CharacterData(char_name=char_name,file_name=file_name,full_path= base_path + full_path,ref_text=ref_text))
            else:
                characters[char_name] = []    
                characters[char_name].append(CharacterData(char_name=char_name,file_name=file_name,full_path= base_path + full_path,ref_text=ref_text))
                
        return characters
    def done(self):
        wandb.finish()

if __name__ == "__main__":
    harness = TestHarness()
    harness.test("Models/LibriTTS/epochs_2nd_00020.pth")

  