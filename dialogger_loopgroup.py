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
import audiosr

device = "cuda"


# set basic vars  - configure these as inputs eventually
split_sentences=True
script_fn = "bg_script_en.txt"
run_number = 7

# read in the script - should be in the following format why not
# lang-code: speaker: line
# en: Jake :I say this line here
                               
file = open(script_fn)
script_dict = []
for line in file:
    fields = line.split(": ")
    print(fields)
    tmp = {}    
    tmp['lang']     = fields[0]
    tmp['speaker']  = fields[1]
    tmp['line']     = fields[2].replace("\n" , '')
    script_dict.append(tmp)

file.close()


speakers = []
for line in script_dict:
    if line['speaker'] not in speakers:
        speakers.append(line['speaker']) 


def get_random_clip():
    files = os.listdir('/home/oem/coding/TTS/TTS/voice_data/randoms')
    voice = random.sample(files, 1)[0]
    voice = os.path.join('/home/oem/coding/TTS/TTS/voice_data/randoms', voice)
    return voice


speaker_ref = {}
for speaker in speakers:
    speaker_ref[speaker] = get_random_clip()

try:
    final_out_dir = 'outputs/loopgroup_' + str(run_number) + '/'
    os.mkdir(final_out_dir)
except:
    print('ehhh')



# Currently loading basic model 
# Add here the xtts_config path
CONFIG_PATH = "recipes/ljspeech/xtts_v2/run/training/XTTS_v2.0_original_model_files/config.json"
CHECKPOINT_DIR = "recipes/ljspeech/xtts_v2/run/training/XTTS_v2.0_original_model_files/"
TOKENIZER_PATH = "recipes/ljspeech/xtts_v2/run/training/XTTS_v2.0_original_model_files/vocab.json"
XTTS_CHECKPOINT = "recipes/ljspeech/xtts_v2/run/training/XTTS_v2.0_original_model_files/model.pth"

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

#clear temp dirs
os.system('rm -r /home/oem/coding/TTS/TTS/outputs/raws2/')
os.mkdir('/home/oem/coding/TTS/TTS/outputs/raws2/')


# for this instance we're only going to make one version per line so we'll constrain it more
line_ind = 0
for entry in script_dict:
    line_ind += 1
    text_prompt = entry['line']
    print(text_prompt)
    lang = entry['lang']
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents( audio_path=speaker_ref[entry['speaker']] )
    
    out = model.inference(
        text_prompt,
        lang,
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.5, 
        top_k=200,
        )
    out_fn = 'outputs/raws2/line_' + str(line_ind) +'.wav'
    torchaudio.save(out_fn, torch.tensor(out["wav"]).unsqueeze(0), 24000)


# flush torch here probably 
del model
torch.cuda.empty_cache()

#def audio_length_test(in_fn, sr=24000):
#    data = librosa.load(in_fn, sr=24000)
#    if len(data[0]) < sr*3:
#        append zeros to array
        
# NOW upsample! 

audiosr_model = audiosr.build_model(model_name="speech", device="cuda")

torch.set_float32_matmul_precision("high")


# gotta keep em in ORDER
def fn_adjuster(val):
    if val<10:
        return '0'+str(val)
    else:
        return str(val)


line_ind = 0

for entry in script_dict:
    line_ind += 1
    print(entry['line'])
    in_fn = 'outputs/raws2/line_' + str(line_ind) +'.wav'
    # audio_length_test(in_fn, sr=24000)            
    waveform = audiosr.super_resolution(
        audiosr_model,
        in_fn,
        seed=42,
        guidance_scale=3.5,
        ddim_steps=50,
        latent_t_per_second=12.8
    )
    # delightfully 32-bit and now at 48 thanks to audiosr
    # kick it up to 96K with librosa
    upsampled = librosa.resample( waveform[0][0], orig_sr=48000 , target_sr=96000 , res_type='soxr_qq')
    # now normalize with pydub....  this will require some I/O and isn't strictly necessary'
    # upsampled = pydub.effects.normalize( upsampled, headroom=10)
    print("upsampled to 32/96 now saving to disk...")
    out_fn = final_out_dir + 'line_' + fn_adjuster(line_ind) + '.wav'
    scipy.io.wavfile.write(filename=out_fn, rate=96000, data=upsampled)



# OK NOW # for each line in the resulting directory, load lines, strip leading and ending silence, normalize and concatenate!

finalout = pydub.AudioSegment.silent(duration=1000) # create 1s of leading silence - we'll append to this
takes = os.listdir(final_out_dir)
takes.sort()

for line_ind in range(1, len(os.listdir(final_out_dir) ) + 1 ):
    tmp = pydub.AudioSegment.from_file( final_out_dir + 'line_' + fn_adjuster(line_ind) +'.wav' )
    tmp = pydub.effects.normalize(tmp, 10)
    #ending_silence = pydub.silence.detect_leading_silence( tmp.reverse() , silence_threshold=-20, chunk_size= 200) 
    #tmp = tmp[0:ending_silence]
    finalout = finalout.append(tmp)

final_out_fn = final_out_dir + 'performance_' + fn_adjuster(run_number)+ '.wav'
finalout.export(out_f=final_out_fn, format='wav')




