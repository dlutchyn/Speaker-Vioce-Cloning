import numpy as np

from encoder import audio
import os


def save_mel_from_wav(wav_path: str, dataset_root_name: str = 'by_book',
                        mels_folder_name: str = 'mels', ext: str = '.wav',
                        min_len_frames: int = 1):
    out_path = wav_path.replace(dataset_root_name, mels_folder_name).replace(ext, '')

    # Load and preprocess the waveform
    wav = audio.preprocess_wav(wav_path)
    if len(wav) == 0:
        return
    # Create the mel spectrogram, discard those that are too short
    frames = audio.wav_to_mel_spectrogram(wav)
    if len(frames) < min_len_frames:
        return
    folder_path = '/'.join(out_path.split('/')[:-1]) + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    np.save(out_path, frames)
    return len(frames)


def get_poems_paths(path_to_speaker):
    paths_to_poem = []
    for name in os.listdir(path_to_speaker):
        path_to_poem = os.path.join(path_to_speaker, name)
        if os.path.isdir(path_to_poem):
            paths_to_poem.append(path_to_poem)
    return paths_to_poem


def get_speakers_paths_names(path_to_all_speakers):
    paths_to_speakers = []
    for name in os.listdir(path_to_all_speakers):
        path_to_speaker = os.path.join(path_to_all_speakers, name)
        if os.path.isdir(path_to_speaker):
            paths_to_speakers.append((path_to_speaker, name))
    return paths_to_speakers


def get_all_poems_paths(dataset_path):
    paths_to_poems_all_with_speaker_name = []
    paths_to_all_speakers = [os.path.join(dataset_path, indicator)
                             for indicator in ['male', 'female']]

    speakers_paths_names = []
    for path_to_all_speakers in paths_to_all_speakers:
        speakers_paths_names_ind = get_speakers_paths_names(path_to_all_speakers)
        speakers_paths_names += speakers_paths_names_ind

    for path_to_speaker, speaker_name in speakers_paths_names:
        paths_to_poems = get_poems_paths(path_to_speaker)
        path_to_poems_with_speaker_names = [(path, speaker_name) for path in paths_to_poems]
        paths_to_poems_all_with_speaker_name += path_to_poems_with_speaker_names
    return paths_to_poems_all_with_speaker_name


def one_poem_uk(root_path, meta_file, speaker_name):
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            text = cols[2]
            text = text[:-1] if text[-1] == '\n' else text
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name})
    return items


def ukrainian_dataset(root_path, meta_file, **kwargs):
    poems_paths_with_speaker_name = get_all_poems_paths(root_path)
    items = []
    for poem_path, speaker_name in poems_paths_with_speaker_name:
        items += one_poem_uk(poem_path, meta_file, speaker_name)
    return items


if __name__ == "__main__":
    root_path = '/home/dlutc/Documents/UCU/Real-Time-Voice-Cloning/' \
               'dataset/uk_UK/by_book/'
    metafile = 'metadata.csv'
    # Test case 1
    # wav_path = '/home/nkusp/UCU_courses/ML_workspace/Real-Time-Voice-Cloning/' \
    #            'datasets/uk_UK/by_book/male/loboda/chorna_rada/wavs/chorna_rada_s000001.wav'
    # save_mel_from_wav(wav_path)

    # Test case 2
    uk = ukrainian_dataset(root_path, metafile)
    print(uk)

    for val in uk:
        wav_path = val['audio_file']
        save_mel_from_wav(wav_path)