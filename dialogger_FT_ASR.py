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


####################### MANUAL ENTRY
# eventually these three should be configured as command line flags but for now...
name        = 'KG'
lang        = 'en'

FT_run      = 4
run_number  = '6_en'

num_takes   = 4
TEMPERATURE = .75

split_sentences         = True
# longer sentences trail off weird, shorter ones do better

script_to_read = 'scripts_to_read/kerry.txt'

# single vs multiple reference audio
single_reference_audio  = False
reference_audio         = 'voice_data/KG_Training_Data_V1.wav'

source_dir = "/home/oem/coding/TTS/TTS/voice_data/" + name + "_all/"


#END OF MANUAL ENTRY
##################################



source_wavs = []
for fn in os.listdir(source_dir):
    source_wavs.append(source_dir + fn)

if len(source_wavs) > 50:
    source_wavs = random.sample(source_wavs, 50)

try:
    final_out_dir = 'outputs/FT' + str(FT_run) + '_' + name + '_results_' + str(run_number) + '/'
    os.mkdir(final_out_dir)
except:
    print('ehhh')



# Add here the xtts_config path
CONFIG_PATH = "recipes/ljspeech/xtts_v2/run/training/" + name + "_FT_v" + str(FT_run) + "/config.json"
CHECKPOINT_DIR = "recipes/ljspeech/xtts_v2/run/training/" + name + "_FT_v" + str(FT_run) + "/"
# Add here the vocab file that you have used to train the model
TOKENIZER_PATH = "recipes/ljspeech/xtts_v2/run/training/" + name + "_FT_v" + str(FT_run) + "/vocab.json"
# Add here the checkpoint that you want to do inference with
XTTS_CHECKPOINT = "recipes/ljspeech/xtts_v2/run/training/" + name + "_FT_v" + str(FT_run) + "/best_model.pth"
# Add here the speaker reference
SPEAKER_REFERENCE = reference_audio

print("Loading model...")
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)

model.load_checkpoint(config, 
        checkpoint_dir=CHECKPOINT_DIR, 
        checkpoint_path=XTTS_CHECKPOINT, 
        vocab_path=TOKENIZER_PATH, 
        use_deepspeed=False)

model.cuda()


print("Computing speaker latents...")
# case based on whether we're using a directory or a single file
if single_reference_audio == True:
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents( audio_path=[reference_audio] )
else:
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents( audio_path=source_wavs )


lines = []
with open(script_to_read,'r') as file:
    for line in file:
        lines.append(line.replace('\n',''))

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
        out = model.inference(
            line,
            lang,
            gpt_cond_latent,
            speaker_embedding,
            temperature=TEMPERATURE, 
            top_k=500,)
        out_fn = 'outputs/raws2/' + name + '_line_' + str(line_ind) +'_v' +str(index) +'.wav'
        torchaudio.save(out_fn, torch.tensor(out["wav"]).unsqueeze(0), 24000)


# flush torch here probably 
del model
torch.cuda.empty_cache()

# TODO Inefficient use until I can encode this into the model itself
# TODO get BigVGAN running at 32 bit depth, higher sample rate???
 
# os.system('python /home/oem/coding/TTS/BigVGAN/inference.py --input_wavs_dir /home/oem/coding/TTS/TTS/outputs/raws/ --output_dir /home/oem/coding/TTS/TTS/outputs/raws2 --checkpoint_file /home/oem/coding/TTS/BigVGAN/bigvgan_24khz_100band/g_05000000.zip')



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

