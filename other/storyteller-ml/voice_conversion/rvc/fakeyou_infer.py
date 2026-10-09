import torch
from fairseq import checkpoint_utils
from scipy.io import wavfile

from vc_infer_pipeline import VC
from my_utils import load_audio
from infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from config import Config


def vc_single(
    hubert_path,
    sid,
    input_audio,
    transpose,
    f0_file,
    f0_method,
    file_index,
    index_rate,
    audio_output_path
):
    global tgt_sr, net_g, vc, hubert_model
    transpose = int(transpose)
    try:
        audio = load_audio(input_audio, 16000)
        times = [0, 0, 0]
        if hubert_model == None:
            load_hubert(hubert_path)
        if_f0 = cpt.get("f0", 1)

        # No idea wtf this is doing
        file_index = (
            file_index.strip(" ")
            .strip('"')
            .strip("\n")
            .strip('"')
            .strip(" ")
            .replace("trained", "added")
        )

        audio_out = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            times,
            transpose,
            f0_method,
            file_index,
            index_rate,
            if_f0,
            f0_file=f0_file,
        )
        print(
            "npy: ", times[0], "s, f0: ", times[1], "s, infer: ", times[2], "s", sep=""
        )
        wavfile.write(
            audio_output_path, tgt_sr, audio_out
        )

        return "Success", (tgt_sr, audio_out)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)

def load_model(model_path):
    global n_spk, tgt_sr, net_g, vc, cpt, hubert_model
    if model_path == []:
        if hubert_model != None:
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr  # ,cpt
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if_f0 = cpt.get("f0", 1)
            if if_f0 == 1:
                net_g = SynthesizerTrnMs256NSFsid(
                    *cpt["config"], is_half=config.is_half
                )
            else:
                net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
        return {"visible": False, "__type__": "update"}
    person = model_path
    print("loading %s" % person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    if if_f0 == 1:
        net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
    else:
        net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    return {"visible": True, "maximum": n_spk, "__type__": "update"}

def load_hubert(hubert_path="hubert_base.pt"):
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [hubert_path],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()



def infer(
    model_path,
    index_path,
    hubert_path,
    audio_input_path,
    audio_output_path,
    index_rate=0.45,
    f0_method="harvest",
    transpose=0,
):

    load_model(model_path)
    vc_single(
        hubert_path, sid=0, input_audio=audio_input_path, transpose=transpose, f0_file=None,
        f0_method=f0_method, file_index=index_path, index_rate=index_rate, audio_output_path=audio_output_path
    )

hubert_model = None
config=Config()
