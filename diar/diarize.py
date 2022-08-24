from os import path
import numpy as np
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm
from subprocess import run
import pickle as pkl
from .cluster import cluster_AHC, cluster_SC


class Diarizer:

    def __init__(self,
                 embed_model='xvec',
                 cluster_method='sc',
                 window=1.5,
                 period=0.75):

        assert embed_model in [
            'xvec', 'ecapa'], "Only xvec and ecapa are supported options"
        assert cluster_method in [
            'ahc', 'sc'], "Only ahc and sc in the supported clustering options"

        if cluster_method == 'ahc':
            self.cluster = cluster_AHC
        elif cluster_method == 'sc':
            self.cluster = cluster_SC

        self.vad_model, self.get_speech_ts = self.setup_VAD()

        run_opts = {"device": "cuda:0" if torch.cuda.is_available() else "cpu"}

        if embed_model == 'xvec':
            self.embed_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
                                                              savedir="pretrained_models/spkrec-xvect-voxceleb",
                                                              run_opts=run_opts)
        elif embed_model == 'ecapa':
            self.embed_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                              savedir="pretrained_models/spkrec-ecapa-voxceleb",
                                                              run_opts=run_opts)

        self.window = window
        self.period = period

    def setup_VAD(self):
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad')

        return model, utils[0]

    def vad(self, signal):
        """
        Runs the VAD model on the signal
        """
        return self.get_speech_ts(signal, self.vad_model)

    def windowed_embeds(self, signal, fs, window=1.5, period=0.75):
        """
        Calculates embeddings for windows across the signal
        window: length of the window, in seconds
        period: jump of the window, in seconds
        returns: embeddings, segment info
        """
        len_window = int(window * fs)
        len_period = int(period * fs)
        len_signal = signal.shape[1]

        # Get the windowed segments
        segments = []
        start = 0
        while start + len_window < len_signal:
            segments.append([start, start+len_window])
            start += len_period

        segments.append([start, len_signal-1])
        embeds = []

        with torch.no_grad():
            for i, j in segments:
                signal_seg = signal[:, i:j]
                seg_embed = self.embed_model.encode_batch(signal_seg)
                embeds.append(seg_embed.squeeze(0).squeeze(0).cpu().numpy())

        return np.array(embeds), np.array(segments)

    def recording_embeds(self, signal, fs, speech_ts):
        """
        Takes signal and VAD output (speech_ts) and produces windowed embeddings
        returns: embeddings, segment info
        """
        all_embeds = []
        all_segments = []

        for utt in tqdm(speech_ts, desc='Utterances', position=0):
            start = utt['start']
            end = utt['end']

            utt_signal = signal[:, start:end]
            utt_embeds, utt_segments = self.windowed_embeds(utt_signal,
                                                            fs,
                                                            self.window,
                                                            self.period)
            all_embeds.append(utt_embeds)
            all_segments.append(utt_segments + start)

        all_embeds = np.concatenate(all_embeds, axis=0)
        all_segments = np.concatenate(all_segments, axis=0)
        return all_embeds, all_segments

    @staticmethod
    def join_segments(cluster_labels, segments, fs):
        """
        Joins up same speaker segments, resolves overlap conflicts
        Uses the midpoint for overlap conflicts
        tolerance allows for very minimally separated segments to be combined
        (in samples)
        """
        assert len(cluster_labels) == len(segments)

        new_segments = [{'start': segments[0][0],
                         'end': segments[0][1],
                         'label': cluster_labels[0]}]

        for l, seg in zip(cluster_labels[1:], segments[1:]):
            start = seg[0]
            end = seg[1]

            protoseg = {'start': seg[0],
                        'end': seg[1],
                        'label': l}

            if start <= new_segments[-1]['end']:
                # If segments overlap
                if l == new_segments[-1]['label']:
                    # If overlapping segment has same label
                    new_segments[-1]['end'] = end
                else:
                    # If overlapping segment has diff label
                    # Resolve by setting new start to midpoint
                    # And setting last segment end to midpoint
                    overlap = new_segments[-1]['end'] - start
                    midpoint = start + overlap//2
                    new_segments[-1]['end'] = midpoint
                    protoseg['start'] = midpoint
                    new_segments.append(protoseg)
            else:
                # If there's no overlap just append
                new_segments.append(protoseg)

        for seg in new_segments:
            seg['start'] = seg['start']/fs
            seg['end'] = seg['end']/fs

        return new_segments

    def diarize(self,
                wav_file,
                n_speakers=2,
                threshold=None,
                silence_tolerance=0.2,
                enhance_sim=True,
                output_file=None):

        recname = path.splitext(path.basename(wav_file))[0]
        converted_wavfile = path.join(path.dirname(
            wav_file), f'{recname}_converted.wav')

        if not path.exists(converted_wavfile):
            print("Converting audio file to single channel WAV using ffmpeg...")
            cmd = f"ffmpeg -y -i {wav_file} -acodec pcm_s16le -ar 16000 -ac 1 {converted_wavfile}"
            run(cmd, shell=True)

        assert path.isfile(
            converted_wavfile), "Couldn't find converted wav file, failed for some reason"
        signal, fs = torchaudio.load(converted_wavfile)

        print('Running VAD...')
        speech_ts = self.vad(signal[0])
        print(f'Splitting by silence found {len(speech_ts)} utterances')
        assert len(speech_ts) >= 1, "Couldn't find any speech during VAD"

        print('Extracting embeddings...')
        embeds, segments = self.recording_embeds(signal, fs, speech_ts)

        print(f'Clustering to {n_speakers} speakers...')
        cluster_labels = self.cluster(embeds, n_clusters=n_speakers,
                                      threshold=threshold, enhance_sim=enhance_sim)

        print('Cleaning up output...')
        cleaned_segments = self.join_segments(cluster_labels, segments, fs)
        cleaned_segments = self.join_samespeaker_segments(cleaned_segments,
                                                          silence_tolerance=silence_tolerance)
        print('Done!')

        if output_file:
            print(f'Saving segments in {output_file}')
            with open(output_file, "wb") as f:
                pkl.dump(cleaned_segments, f)

        return cleaned_segments

    @staticmethod
    def join_samespeaker_segments(segments, silence_tolerance=0.5):
        """
        Join up segments that belong to the same speaker, 
        even if there is a duration of silence in between them.
        If the silence is greater than silence_tolerance, does not join up
        """
        new_segments = [segments[0]]

        for seg in segments[1:]:
            if seg['label'] == new_segments[-1]['label']:
                if new_segments[-1]['end'] + silence_tolerance >= seg['start']:
                    new_segments[-1]['end'] = seg['end']
                else:
                    new_segments.append(seg)
            else:
                new_segments.append(seg)
        return new_segments
