import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    # "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map="cpu",
    dtype=torch.bfloat16,  # bfloat16 is often sketchy on MPS
    attn_implementation="sdpa",  # or just omit this argument
)

crofty_instruct = "Male UK broadcast voice. Keep British vowel shapes. Pronounce 'bath, grass, chance' with broad /ɑː/. Non-rhotic: drop the R in 'car, driver'. Crisp T consonants. No American R-coloring. Fast, excited F1 commentary cadence."

f1_style_ref = """
Welcome back to the circuit. Conditions are stable. Track temperature is rising, and grip is improving lap by lap.The cars are now settled into race pace, with small adjustments on brake bias and differential to manage tire life. We're watching the gap ahead, the gap behind, and the tire delta. It's not just speed, it's timing.The next few laps will decide whether this is a one stop or a 2 stop race. Across the line now, one minute, 38.6, sect... one is purple. Sector 2 is ready.The gap is 1.. 3 seconds. DRS range is 12nd. So this is right on the edge.He's gaining 2 tents through the fast section, but losing it under traction. Tyres, soft, medium, hard. New verses used, fresh rubber makes a difference on the exit.He's getting closer now. This is the lap where it starts to happen. He's tucked into the slipstream.He's inching forward and he's committing early. Later on the brakes, front end bites, rear steps out, still on alongside. Wheel to wheel in the corner and he's got it done.That's a decisive move. That's a lockup. That's a big lockup.Smoke off the front tires. He's wide. He's over the curb, and that's cost him momentum.The car snaps, he catches it. That was nearly a spin. Yellow flags in sector two. something has happened ahead.Copy, understood. Keep it clean, hit your marks, tire management now. Dont slide knit the rear.Mode 7 on the straight, we're boxing this lap. confirm, box, box. Final lap. This is it.He's pushing to the limit. No mistakes, no mercy. Out of the glass corner now, flat to the line.And that is a brilliant dive.
"""

crofty_ref = """
mercedes threw everything at him today Charles Leclerc go brilliantly he won in spa he wins in monza the sun is out the smiles will be out at 18 years and 227 days old max verstappen wins a formula one grand prix the man that woke up this morning on the verge of an historic 92nd win in formula one records are there to be broken said michael schumacher the record is and they are gonna win in 2020 as alpha terry pierre gasly wins the italian grand prix the man who was in last place at the end of lap one comes home to win the sakira grand prix sergio perez wow what a race today in monte carlo it's redemption day for daniel ricardo he wins the monaco grand prix and he will celebrate that for a long long time to come we'll fail now who's going to be last on the breaks leclair has that inside line perez goes off the track comes the chicane golf goes leclair through goes they have shared a brilliant championship battle but the championship could only be one by one and it's going dutch in 2021 max verstappen for the first time ever is champion of the world to the top and third ready on and magnuson has made up places in the start but here comes sebastian vennel he's there connect with lewis hamilton"""

output_text = """I am an Israeli emigrant currently living in the UK and want to learn arabic for interest reasons and to possibly read the Quran in arabic eventually (I am not Muslim), since most Arabs in israel speak Palestinian arabic, I'd like to learn Levantine Arabic, what are the best resources for this? And any advantage I could get from the Hebrew?

Thanks

Edit: I don't want this to be political but I know that it's a hot topic, so fyi I am very against the Israeli government in its conduct in Gaza and the West Bank, part of my reason for learning Arabic is to understand first hand recounts of Palestinians and gain a deeper understanding into the conflict"""

output_text = "Welcome back to the circuit. Conditions are stable, track temperature is rising, and grip is improving lap by lap. The cars are now settled into race pace, with small adjustments on brake bias and differential to manage tire life, AND HE'S DONE IT!!! Charles Leclerc edges Hamilton, AND THERE'S CONTACT!!! A MASSIVE CRASH. Unbelievable."

yasmin_ref = """
Hi, I'm going to read a short sample to the recording sounds like me in real life. Today I woke up a little earlier than usual, made a drink, and checked my phone for messages. I'm speaking at a normal pace and I'll also include a few questions.What time is it right now? Did you get my last message? Are we still meeting later?If you're lost, call me and tell me exactly where you are. Here are some numbers. 37, 14, 19, 20, 42, 58, 101, 250, 999 and 2026. And sometimes in dates, 815, 1230, 645 PM, Monday, Wednesday, 1st of March, 22nd of January.I'll spell a few words. A L P-H-A, B-R-A-V-O, C-H-A-R-L-I-E, D-E-L-T-A, email, message, password, account, confirmation, delivery, address, receipt. Now I sound a bit more excited.I can't believe that actually worked. They're so satisfying. And now more serious.Please listen carefully. This is important. Finally, I'm going to read one longer sentence without rushing, and I'll pause naturally when it feels right.In the future, I want this recording to sound clean, clear, and realistic. So I'm speaking comfortably and keeping the microphone steady."""

aadwit_ref = """
Tianmin Square, protest led by students known in China as the June 4th incident, were held in Tianmin Square in Beijing. China from 15th April to 4th June, 1989. After weeks of unsuccessful attempts between the demonstrators and the Chinese government to find a peaceful resolution, the Chinese government deployed troops to occupy the square on the night of 3rd June, what is referred to as the Tian Min Square Massacre. The events are sometimes called the 89 democracy movement. The Chanman Square incident, or the Sandman Square uprising."""

# wavs, sr = model.generate_voice_design(
#     text=output_text,
#     language="English",
#     instruct=crofty_instruct,
# )
# sf.write("output_custom_voice.wav", wavs[0], sr)

wavs, sr = model.generate_voice_clone(
    text=output_text,
    language="English",
    ref_audio="voices/f1_style.wav",
    ref_text=f1_style_ref,
)
sf.write("f1_style_output.wav", wavs[0], sr)
