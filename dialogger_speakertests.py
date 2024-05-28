import torch
import numpy as np
import scipy
import librosa
import subprocess
import os , glob , random
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import pydub
device = "cuda"

#from BigVGAN import inference as BigVGAN_inference
#from TTS.api import TTS
#import voicefixer
lingoes = ['en', 'es', 'fr', 
    'de', 'it', 'pt', 'pl', 
    'tr', 'ru', 'nl', 'cs', 
    'ar', 'zh-cn', 'hu', 
    'ko', 'ja', 'hi']

speakers= ['Claribel Dervla'], 'Daisy Studious', 'Gracie Wise', 'Tammie Ema', 'Alison Dietlinde', 'Ana Florence', 'Annmarie Nele', 'Asya Anara', 'Brenda Stern', 'Gitta Nikolina', 'Henriette Usha', 'Sofia Hellen', 'Tammy Grit', 'Tanja Adelina', 'Vjollca Johnnie', 'Andrew Chipper', 'Badr Odhiambo', 'Dionisio Schuyler', 'Royston Min', 'Viktor Eka', 'Abrahan Mack', 'Adde Michal', 'Baldur Sanjin', 'Craig Gutsy', 'Damien Black', 'Gilberto Mathias', 'Ilkin Urbano', 'Kazuhiko Atallah', 'Ludvig Milivoj', 'Suad Qasim', 'Torcull Diarmuid', 'Viktor Menelaos', 'Zacharie Aimilios', 'Nova Hogarth', 'Maja Ruoho', 'Uta Obando', 'Lidiya Szekeres', 'Chandra MacFarland', 'Szofi Granger', 'Camilla Holmström', 'Lilya Stainthorpe', 'Zofija Kendrick', 'Narelle Moon', 'Barbora MacLean', 'Alexandra Hisakawa', 'Alma María', 'Rosemary Okafor', 'Ige Behringer', 'Filip Traverse', 'Damjan Chapman', 'Wulf Carlevaro', 'Aaron Dreschner', 'Kumar Dahl', 'Eugenio Mataracı', 'Ferran Simen', 'Xavier Hayasaka', 'Luis Moray', 'Marcos Rudaski']

####################### MANUAL ENTRY

# eventually these three should be configured as command line flags but for now...



script_to_read = 'scripts_to_read/kerry.txt'
num_takes   = 3
TEMPERATURE = .75
split_sentences         = True
lang        = 'en'


print("Loading model...")
config = XttsConfig()
#config.load_json(CONFIG_PATH)
#model = Xtts.init_from_config(config)
#
#model.load_checkpoint(config, 
#        checkpoint_dir=CHECKPOINT_DIR, 
#        checkpoint_path=XTTS_CHECKPOINT, 
#        vocab_path=TOKENIZER_PATH, 
#        use_deepspeed=False)
#
#model.cuda()
#





for name in speakers:
    model = Xtts.init_from_config(config)
    model.cuda()
    # longer sentences trail off weird, shorter ones do better
    # case based on whether we're using a directory or a single file
    lines = []
    with open(script_to_read,'r') as file:
        for line in file:
            lines.append(line.replace('\n',''))
    final_out_dir = 'outputs/speakertests/' + name + '/'
    
    os.system('rm -r /home/oem/coding/TTS/TTS/outputs/raws/')
    os.mkdir('/home/oem/coding/TTS/TTS/outputs/raws/')
    os.system('rm -r /home/oem/coding/TTS/TTS/outputs/raws2/')
    os.mkdir('/home/oem/coding/TTS/TTS/outputs/raws2/')
    
    line_ind = 0
    for line in lines:
        line_ind += 1
        text_prompt = line
        for index in range(1, num_takes ):
            # ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
            print(line + ' '+ str(line_ind))
            out = model.synthesize(
                text=line,
                config=config,
                langguage=lang,
                speaker_id=name
            )
            out_fn = 'outputs/raws2/' + name + '_line_' + str(line_ind) +'_v' +str(index) +'.wav'
            torchaudio.save(out_fn, torch.tensor(out["wav"]).unsqueeze(0), 24000)
    
    # flush torch here probably 
    del model
    torch.cuda.empty_cache()
    
    import audiosr
    #using AUDIOSR speech model cause it rocks
    audiosr_model = audiosr.build_model(model_name="speech", device="cuda")
    torch.set_float32_matmul_precision("high")
    
    line_ind = 0
    for line in lines:
        line_ind += 1
        text_prompt = line
        print(line )
        for index in range(1, num_takes ):
            in_fn = 'outputs/raws2/' + name + '_line_' + str(line_ind) +'_v' +str(index) +'_generated.wav'
            in_fn = 'outputs/raws2/' + name + '_line_' + str(line_ind) +'_v' +str(index) +'.wav'
            # audio_length_test(in_fn, sr=24000)            
            waveform = audiosr.super_resolution(
                audiosr_model,
                in_fn,
                seed=42,
                guidance_scale=3.5,
                ddim_steps=50,
                latent_t_per_second=12.8
            )
            # what format is waveform - np . 32!
            upsampled = librosa.resample( waveform[0][0], orig_sr=48000 , target_sr=96000 , res_type='soxr_qq')
            # now normalize with pydub
            # upsampled = pydub.effects.normalize( upsampled, headroom=10)
            print("upsampled to 32/96 now saving to disk...")
            out_fn = final_out_dir + name + '_line_' + str(line_ind) +'_v' +str(index) +'.wav'
            scipy.io.wavfile.write(filename=out_fn, rate=96000, data=upsampled)
    
    print('cleaning...')
    os.system('rm -r /home/oem/coding/TTS/TTS/outputs/raws/')
    os.system('rm -r /home/oem/coding/TTS/TTS/outputs/raws2/')

