from datetime import timedelta, time
import pickle as pkl
from webvtt import read
from subprocess import run, DEVNULL, PIPE
from tqdm import tqdm
import os
from shutil import rmtree, copy
from diar.diarize import Diarizer
from inaSpeechSegmenter import Segmenter
from glob import glob
import warnings
import librosa

warnings.simplefilter(action='ignore', category=FutureWarning)


def download_subs(url: str,
                  lang='en',
                  subs_format='vtt',
                  output_name='output',
                  download_audio=True):
    """Download subtitles"""
    skip_download = '--skip-download' if not download_audio else ''

    cmd = f'youtube-dl --write-sub \
            --sub-format {subs_format} \
            --sub-lang {lang} \
            {skip_download} \
            --extract-audio --audio-format wav \
            -o {output_name}.tmp {url}'

    run(cmd, shell=True)


def extract_subs(subs_file):
    """
    Extract subtitles from file
    """
    return [{'start': time_to_obj(sub.start),
            'end': time_to_obj(sub.end),
             'text': sub.text}
            for sub in read(subs_file)]


def get_dataset_duration(dataset_dir: str) -> float:
    """Get dataset duration in minutes"""
    return sum(librosa.get_duration(filename=f) for f in glob(f'{dataset_dir}/*.wav')) / 60


def has_subs(url: str, lang='en') -> bool:
    """Check whether url contains subs in a language we need"""
    cmd = f'youtube-dl --list-sub {url}'
    out = run(cmd, shell=True, stdout=PIPE).stdout.decode()

    subs_exist = False
    for line in out.split('\n'):
        if line.startswith('Available subtitles for'):
            subs_exist = True
        if subs_exist and line.startswith(lang):
            return True
    return False


def time_to_obj(timestr: str):
    """Converts ISO string to timedelta object"""
    timeobj = time.fromisoformat(timestr)
    return timedelta(
        hours=timeobj.hour,
        minutes=timeobj.minute,
        seconds=timeobj.second,
        microseconds=timeobj.microsecond).total_seconds()


def clean_text(text: str) -> str:
    """
    Stips text of unneded characters.
    This function is expected to grow as we run into new cases
    """
    return text.replace('\n', ' ') + ' '


def align(voice_segments, text_segments, n_speakers, min_sample_len=3):
    """Align audio and text segments"""
    aligned_segments = {i: [] for i in range(n_speakers)}

    for v_seg in tqdm(voice_segments):
        cls = v_seg['label']
        st = v_seg['start']
        ed = v_seg['end']
        if (ed-st) >= min_sample_len:
            text = ''

            for sub in text_segments:
                if sub['end'] >= st and sub['start'] <= ed:
                    text = text + clean_text(sub['text'])
                elif sub['start'] > ed:
                    break

            seg = (text[:-1], st, ed)
            aligned_segments[cls].append(seg)

    return aligned_segments


def save_segments(aligned_segments, audio_file,
                  output_dir="samples",
                  output_format="folder"):
    """
    Save aligned segments. If output_format is folder, then
    segments will be saved as separate files in output_dir.
    If output_format is metadata, then
    segments will be saved as entries in metadata.csv
    """

    assert output_format in {"folder", "metadata"}
    os.makedirs(output_dir, exist_ok=True)

    if output_format == "folder":
        for cls, seg in aligned_segments.items():
            for i, (text, st, ed) in enumerate(seg):
                sample_name = f'{output_dir}/{cls}_{i}'
                with open(sample_name+'.txt', 'w') as f:
                    f.write(text)
                cmd = f"ffmpeg -i {audio_file} -ss {st} -to {ed} {sample_name}.wav"
                run(cmd, shell=True, stderr=DEVNULL)

    else:
        copy(audio_file, output_dir+'/input.wav')
        with open(output_dir+'/metadata.csv', 'w') as f:
            for cls, seg in aligned_segments.items():
                for text, st, ed in seg:
                    line = '|'.join(map(str, (cls, st, ed, text)))+'\n'
                    f.write(line)


def metadata_find_noise(output_dir, audio_file):
    """Produce noise_segments.txt file that lists segments with noise
    Can be used only when dataset is saved as metadata"""
    segments = Segmenter()(audio_file)

    with open(output_dir+'/noise_segments.txt', 'w') as f:
        for (cls, st, ed) in segments:
            if cls == 'noise':
                f.write(f'{st}|{ed}\n')


def delete_noise_segs(output_dir):
    """Delete noise segments from dataset
    Can be used only when dataset is saved as multpiple files."""
    seg = Segmenter()
    for sample in glob(output_dir+'/*.wav'):
        out = seg(sample)
        if any(i[0] == 'noise' for i in out):
            print(sample, out)
            os.remove(sample)
            os.remove(sample.replace('.wav', '.txt'))


def align_from_files(audio_file, n_speakers, subs_file, dia_file,
                     output_format="folder"):
    """
    Auxillary function used mostly for testing
    """
    with open(dia_file, 'rb') as f:
        voice_segments = pkl.load(f)

    subs = extract_subs(subs_file)

    aligned_segments = align(voice_segments, subs, n_speakers)
    save_segments(aligned_segments, audio_file, output_format=output_format)


def make_dataset_from_url(url,
                          n_speakers,
                          lang='en',
                          output_dir='samples',
                          temp_dir='tmp',
                          cleanup=False,
                          output_format="folder",
                          remove_noise_segments=True):
    """
    Create dataset from URL
    Keyword arguments:
    url -- url of youtube video do parse
    n_speakers -- amount of speakers in video
    lang -- language of subtitles
    output_dir -- directory for dataset
    temp_dir -- directory for temporary files
    cleanup -- whether to clean temp_dir after making dataset
    remove_noise_segments -- whether to remove segments with noise
    """

    os.makedirs(temp_dir, exist_ok=True)

    if has_subs(url, lang):

        download_subs(url, lang=lang, output_name=temp_dir+'/subs')

        audio_file = temp_dir+'/subs.wav'
        subs_file = temp_dir+f'/subs.tmp.{lang}.vtt'

        voice_segments = Diarizer().diarize(audio_file, n_speakers=n_speakers)
        subs = extract_subs(subs_file)

        aligned_segments = align(voice_segments, subs, n_speakers)

        save_segments(aligned_segments, audio_file, output_dir, output_format)

        if remove_noise_segments:
            if output_format == "folder":
                delete_noise_segs(output_dir)
            else:
                metadata_find_noise(output_dir, audio_file)

        if cleanup:
            rmtree(temp_dir)
    else:
        print(f'{url} has no subs in {lang}')
