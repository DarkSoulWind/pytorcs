import csv
import os
import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel
from tqdm import tqdm

VOICE_DESIGN = True  # false by default, cloning voice
VOICE = "voices/crofty.wav"  # voice to clone
OUT = "vd_output_1.wav"
LOCK_VOICE_DESIGN = True  # keep one generated voice across all lines
ENERGY_BOOST = True  # keep race-call intensity instead of flat narration
SHOUT_BOOST = True  # increase projection/intensity for commentator delivery
OUTPUT_GAIN_DB = 4.0  # louder final mix; limiter below prevents harsh clipping
OFFLINE_MODELS_DIR = os.getenv(
    "QWEN_TTS_MODELS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "qwen_tts"),
)
VOICE_SEED_TEXT = (
    "LIGHTS OUT AND AWAY WE GO!!! VERSTAPPEN GETS A BRILLIANT LAUNCH, LECLERC DIVES TO THE INSIDE, "
    "THEY ARE WHEEL TO WHEEL THROUGH TURN ONE, THIS IS ABSOLUTE CHAOS, LISTEN TO THIS CROWD!"
)
VOICE_DESIGN_INSTRUCT = (
    "Adult male British Formula 1 lead commentator, baritone timbre and chest resonance. "
    "Live race intensity at 10/10 energy: ecstatic, urgent, emotionally charged, and punchy. "
    "Projected stadium-level delivery with a raised chest voice, forceful attack, and controlled rasp. "
    "Frequently shout key race moments while staying intelligible. "
    "Rapid cadence with sharp dynamic contrast, big pitch lifts on attacks and overtakes, "
    "hard emphasis on names and decisive verbs, frequent exclamation-style delivery, natural "
    "surges of adrenaline, audible excitement, and brief emotional breaks on huge moments. "
    "breath noise under pressure. Never flat, never sleepy, never corporate TTS, never calm "
    "audiobook narration. British pronunciation with broad /a:/ in bath/grass/chance, non-rhotic "
    "car/driver, no American R-coloring. Sound like a peak-moment live broadcast from trackside."
)

"""
NOTES 

It's literally taking forever when I use cpu, idk what the heck is going on bruh.

For high energy moments, turn the text into capitals
"""


def silence_pad_token_warning(tts_model):
    gen_cfg = getattr(tts_model, "generation_config", None)
    if gen_cfg is not None and gen_cfg.pad_token_id is None:
        gen_cfg.pad_token_id = gen_cfg.eos_token_id


f1_style_ref = """
Welcome back to the circuit. Conditions are stable. Track temperature is rising, and grip is improving lap by lap.The cars are now settled into race pace, with small adjustments on brake bias and differential to manage tire life. We're watching the gap ahead, the gap behind, and the tire delta. It's not just speed, it's timing.The next few laps will decide whether this is a one stop or a 2 stop race. Across the line now, one minute, 38.6, sect... one is purple. Sector 2 is ready.The gap is 1.. 3 seconds. DRS range is 12nd. So this is right on the edge.He's gaining 2 tents through the fast section, but losing it under traction. Tyres, soft, medium, hard. New verses used, fresh rubber makes a difference on the exit.He's getting closer now. This is the lap where it starts to happen. He's tucked into the slipstream.He's inching forward and he's committing early. Later on the brakes, front end bites, rear steps out, still on alongside. Wheel to wheel in the corner and he's got it done.That's a decisive move. That's a lockup. That's a big lockup.Smoke off the front tires. He's wide. He's over the curb, and that's cost him momentum.The car snaps, he catches it. That was nearly a spin. Yellow flags in sector two. something has happened ahead.Copy, understood. Keep it clean, hit your marks, tire management now. Dont slide knit the rear.Mode 7 on the straight, we're boxing this lap. confirm, box, box. Final lap. This is it.He's pushing to the limit. No mistakes, no mercy. Out of the glass corner now, flat to the line.And that is a brilliant dive.
"""

crofty_ref = """
mercedes threw everything at him today Charles Leclerc go brilliantly he won in spa he wins in monza the sun is out the smiles will be out at 18 years and 227 days old max verstappen wins a formula one grand prix the man that woke up this morning on the verge of an historic 92nd win in formula one records are there to be broken said michael schumacher the record is and they are gonna win in 2020 as alpha terry pierre gasly wins the italian grand prix the man who was in last place at the end of lap one comes home to win the sakira grand prix sergio perez wow what a race today in monte carlo it's redemption day for daniel ricardo he wins the monaco grand prix and he will celebrate that for a long long time to come we'll fail now who's going to be last on the breaks leclair has that inside line perez goes off the track comes the chicane golf goes leclair through goes they have shared a brilliant championship battle but the championship could only be one by one and it's going dutch in 2021 max verstappen for the first time ever is champion of the world to the top and third ready on and magnuson has made up places in the start but here comes sebastian vennel he's there connect with lewis hamilton"""

output_text = """I am an Israeli emigrant currently living in the UK and want to learn arabic for interest reasons and to possibly read the Quran in arabic eventually (I am not Muslim), since most Arabs in israel speak Palestinian arabic, I’d like to learn Levantine Arabic, what are the best resources for this? And any advantage I could get from the Hebrew?

Thanks

Edit: I don’t want this to be political but I know that it’s a hot topic, so fyi I am very against the Israeli government in its conduct in Gaza and the West Bank, part of my reason for learning Arabic is to understand first hand recounts of Palestinians and gain a deeper understanding into the conflict"""

output_text = "Welcome back to the circuit. Conditions are stable, track temperature is rising, and grip is improving lap by lap. The cars are now settled into race pace, with small adjustments on brake bias and differential to manage tire life, AND HE'S DONE IT!!! Charles Leclerc edges Hamilton, AND THERE'S CONTACT!!! A MASSIVE CRASH. Unbelievable."

yasmin_ref = """
Hi, I'm going to read a short sample to the recording sounds like me in real life. Today I woke up a little earlier than usual, made a drink, and checked my phone for messages. I'm speaking at a normal pace and I'll also include a few questions.What time is it right now? Did you get my last message? Are we still meeting later?If you're lost, call me and tell me exactly where you are. Here are some numbers. 37, 14, 19, 20, 42, 58, 101, 250, 999 and 2026. And sometimes in dates, 815, 1230, 645 PM, Monday, Wednesday, 1st of March, 22nd of January.I'll spell a few words. A L P-H-A, B-R-A-V-O, C-H-A-R-L-I-E, D-E-L-T-A, email, message, password, account, confirmation, delivery, address, receipt. Now I sound a bit more excited.I can't believe that actually worked. They're so satisfying. And now more serious.Please listen carefully. This is important. Finally, I'm going to read one longer sentence without rushing, and I'll pause naturally when it feels right.In the future, I want this recording to sound clean, clear, and realistic. So I'm speaking comfortably and keeping the microphone steady."""

aadwit_ref = """
Tianmin Square, protest led by students known in China as the June 4th incident, were held in Tianmin Square in Beijing. China from 15th April to 4th June, 1989. After weeks of unsuccessful attempts between the demonstrators and the Chinese government to find a peaceful resolution, the Chinese government deployed troops to occupy the square on the night of 3rd June, what is referred to as the Tian Min Square Massacre. The events are sometimes called the 89 democracy movement. The Chanman Square incident, or the Sandman Square uprising."""


def load_commentary_events(csv_path):
    events = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        has_new_schema = {"start_time", "text"}.issubset(fieldnames)
        has_old_schema = {"timestamp", "commentary"}.issubset(fieldnames)
        if not has_new_schema and not has_old_schema:
            raise ValueError(
                f"{csv_path} must contain either ['start_time','end_time','text'] "
                f"or ['timestamp','commentary']; got {reader.fieldnames}"
            )

        for i, row in enumerate(reader, start=2):
            text = (
                (row.get("text") if has_new_schema else row.get("commentary")) or ""
            ).strip()
            ts_raw = (
                (row.get("start_time") if has_new_schema else row.get("timestamp"))
                or ""
            ).strip()
            if not text:
                continue
            if not ts_raw:
                raise ValueError(f"Missing timestamp at {csv_path}:{i}")
            try:
                timestamp_sec = float(ts_raw)
            except ValueError as e:
                raise ValueError(
                    f"Invalid timestamp '{ts_raw}' at {csv_path}:{i}"
                ) from e
            if timestamp_sec < 0:
                raise ValueError(f"Negative timestamp '{ts_raw}' at {csv_path}:{i}")
            events.append((timestamp_sec, text))

    events.sort(key=lambda x: x[0])
    return events


# Optional race ambience bed
AMBIENCE_FILE = "assets/f1_crowd_engine_24k_mono.wav"  # mono/stereo wav
AMBIENCE_DB = -10.0  # keep low so speech stays clear
DUCK_STRENGTH = 0.7  # lower ambience while commentator is speaking


def rms_envelope(x, win=2048):
    x2 = x * x
    k = np.ones(win, dtype=np.float32) / win
    return np.sqrt(np.convolve(x2, k, mode="same") + 1e-9)


def to_mono(x):
    if x.ndim == 2:
        return x.mean(axis=1)
    return x


def resample_linear(x, src_sr, dst_sr):
    if src_sr == dst_sr:
        return x
    if x.shape[0] == 0:
        return x
    ratio = float(dst_sr) / float(src_sr)
    dst_len = int(round(x.shape[0] * ratio))
    if dst_len <= 1:
        return np.zeros(max(dst_len, 1), dtype=np.float32)
    src_idx = np.arange(x.shape[0], dtype=np.float32)
    dst_idx = np.linspace(0, x.shape[0] - 1, num=dst_len, dtype=np.float32)
    return np.interp(dst_idx, src_idx, x).astype(np.float32)


def create_tts_model():
    model_dir = (
        os.path.join(OFFLINE_MODELS_DIR, "Qwen3-TTS-12Hz-1.7B-VoiceDesign")
        if VOICE_DESIGN
        else os.path.join(OFFLINE_MODELS_DIR, "Qwen3-TTS-12Hz-1.7B-Base")
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Offline model not found at '{model_dir}'. "
            "Download it first with huggingface-cli."
        )
    print(f"Loading model at {model_dir}")
    tts_model = Qwen3TTSModel.from_pretrained(
        model_dir,
        # device_map="cpu",  # 1m 51s on mps
        # dtype=torch.float16,
        device_map="mps",
        dtype=torch.bfloat16,  # bfloat16 is often sketchy on MPS
        local_files_only=True,
        # attn_implementation="sdpa",  # or just omit this argument
    )
    silence_pad_token_warning(tts_model)
    return tts_model


def setup_locked_voice_seed(model):
    if not (VOICE_DESIGN and LOCK_VOICE_DESIGN):
        return None, None, None

    base_model_dir = os.path.join(OFFLINE_MODELS_DIR, "Qwen3-TTS-12Hz-1.7B-Base")
    if not os.path.isdir(base_model_dir):
        raise FileNotFoundError(
            f"Offline model not found at '{base_model_dir}'. "
            "Download it first with huggingface-cli."
        )
    clone_model = Qwen3TTSModel.from_pretrained(
        base_model_dir,
        device_map="mps",
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    silence_pad_token_warning(clone_model)
    seed_wavs, seed_sr = model.generate_voice_design(
        text=VOICE_SEED_TEXT,
        language="English",
        instruct=VOICE_DESIGN_INSTRUCT,
        do_sample=True,
        temperature=(
            1.18 if ENERGY_BOOST and SHOUT_BOOST else (1.12 if ENERGY_BOOST else 0.9)
        ),
        top_p=0.97 if ENERGY_BOOST else 1.0,
    )
    return clone_model, seed_wavs[0], seed_sr


def generate_line_audio(model, clone_model, seed_audio, seed_sr, line):
    if VOICE_DESIGN:
        if LOCK_VOICE_DESIGN:
            return clone_model.generate_voice_clone(
                text=line,
                language="English",
                ref_audio=(seed_audio, seed_sr),
                ref_text=VOICE_SEED_TEXT,
                x_vector_only_mode=False,  # include style/prosody from the energetic seed
                do_sample=True,
                temperature=(
                    1.15
                    if ENERGY_BOOST and SHOUT_BOOST
                    else (1.08 if ENERGY_BOOST else 0.9)
                ),
                top_p=0.97 if ENERGY_BOOST else 1.0,
            )
        return model.generate_voice_design(
            text=line,
            language="English",
            instruct=VOICE_DESIGN_INSTRUCT,
        )
    return model.generate_voice_clone(
        text=line,
        language="English",
        ref_audio=VOICE,
        ref_text=crofty_ref,
    )


def synthesize_commentary_events_stream(
    model, commentary_events, clone_model=None, seed_audio=None, seed_sr=None
):
    all_events_wavs = []
    sr = None
    total_lines = len(commentary_events)

    with torch.inference_mode():
        for i, (timestamp_sec, line) in enumerate(
            tqdm(commentary_events, total=total_lines)
        ):
            wavs, line_sr = generate_line_audio(
                model, clone_model, seed_audio, seed_sr, line
            )
            if sr is None:
                sr = line_sr
            elif line_sr != sr:
                raise ValueError(f"Sample rate changed: {sr} -> {line_sr}")

            line_wav = (
                wavs[0].cpu().numpy() if isinstance(wavs[0], torch.Tensor) else wavs[0]
            )
            all_events_wavs.append((timestamp_sec, line_wav))
            yield {
                "stage": "audio_generation",
                "status": "progress",
                "generated_lines": i + 1,
                "total_lines": total_lines,
                "timestamp": f"{timestamp_sec:.3f}",
            }

    if not all_events_wavs:
        raise ValueError("No non-empty commentary lines found in commentary.csv")
    yield {
        "stage": "audio_generation",
        "status": "synthesis_done",
        "generated_lines": total_lines,
        "total_lines": total_lines,
        "all_events_wavs": all_events_wavs,
        "sample_rate": sr,
    }


def synthesize_commentary_events(
    model, commentary_events, clone_model=None, seed_audio=None, seed_sr=None
):
    all_events_wavs = []
    sr = None
    for event in synthesize_commentary_events_stream(
        model=model,
        commentary_events=commentary_events,
        clone_model=clone_model,
        seed_audio=seed_audio,
        seed_sr=seed_sr,
    ):
        if event.get("status") == "synthesis_done":
            all_events_wavs = event.get("all_events_wavs", [])
            sr = event.get("sample_rate")
    return all_events_wavs, sr


def build_timeline(all_events_wavs, sr):
    event_buffers = []
    for idx, (timestamp_sec, line_wav) in enumerate(all_events_wavs):
        start_idx = int(round(timestamp_sec * sr))
        clip = line_wav.astype(np.float32)
        if idx + 1 < len(all_events_wavs):
            next_start_idx = int(round(all_events_wavs[idx + 1][0] * sr))
            allowed_len = max(0, next_start_idx - start_idx)
            clip = clip[:allowed_len]
        event_buffers.append((start_idx, clip))

    max_end = 0
    for start_idx, clip in event_buffers:
        max_end = max(max_end, start_idx + clip.shape[-1])

    full_wav = np.zeros(max_end, dtype=np.float32)
    for start_idx, clip in event_buffers:
        if clip.shape[-1] == 0:
            continue
        end_idx = start_idx + clip.shape[-1]
        full_wav[start_idx:end_idx] = clip
    return full_wav


def apply_output_limiter(wav, output_gain_db=OUTPUT_GAIN_DB):
    gain = 10 ** (output_gain_db / 20.0)
    wav = wav * gain
    wav = np.tanh(wav * 1.1) / np.tanh(1.1)
    peak = float(np.max(np.abs(wav)))
    if peak > 0.98:
        wav = wav * (0.98 / peak)
    return wav


def add_ambience(full_wav, sr):
    amb, amb_sr = sf.read(AMBIENCE_FILE, dtype="float32")
    amb = to_mono(amb)
    if amb_sr != sr:
        amb = resample_linear(amb, amb_sr, sr)

    need = full_wav.shape[0]
    if amb.shape[0] < need:
        reps = int(np.ceil(need / amb.shape[0]))
        amb = np.tile(amb, reps)
    amb = amb[:need]

    amb_gain = 10 ** (AMBIENCE_DB / 20.0)
    amb = amb * amb_gain

    voice_env = rms_envelope(full_wav, win=2048)
    voice_env = voice_env / (np.max(voice_env) + 1e-9)
    duck = 1.0 - (DUCK_STRENGTH * voice_env)
    mixed = full_wav + (amb * duck.astype(np.float32))
    return apply_output_limiter(mixed, output_gain_db=0.0)


def run_pipeline_stream(commentary_csv="commentary.csv", out_path=OUT):
    model = create_tts_model()
    clone_model, seed_audio, seed_sr = setup_locked_voice_seed(model)

    commentary_events = load_commentary_events(commentary_csv)
    if not commentary_events:
        raise ValueError(f"No valid commentary rows found in {commentary_csv}")

    all_events_wavs = []
    sr = None
    for event in synthesize_commentary_events_stream(
        model=model,
        commentary_events=commentary_events,
        clone_model=clone_model,
        seed_audio=seed_audio,
        seed_sr=seed_sr,
    ):
        if event.get("status") == "synthesis_done":
            all_events_wavs = event.get("all_events_wavs", [])
            sr = event.get("sample_rate")
        else:
            yield event

    full_wav = build_timeline(all_events_wavs, sr)
    full_wav = apply_output_limiter(full_wav)
    full_wav = add_ambience(full_wav, sr)

    sf.write(out_path, full_wav, sr)
    print(f"Audio commentary output to {out_path}")
    yield {
        "stage": "audio_generation",
        "status": "done",
        "generated_lines": len(commentary_events),
        "total_lines": len(commentary_events),
        "audio_path": out_path,
    }


def run_pipeline(commentary_csv="commentary.csv", out_path=OUT):
    for _ in run_pipeline_stream(commentary_csv=commentary_csv, out_path=out_path):
        pass


if __name__ == "__main__":
    run_pipeline()
