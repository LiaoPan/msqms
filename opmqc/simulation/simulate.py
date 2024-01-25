# -*- coding: utf-8 -*-
"""<Explain your Codes>"""
# Ref from BrainMagic Project.

def create_fake_meg() -> mne.io.RawArray:
    """Creates a fake meg with random noise in it
    """
    n_channels, n_times = 273 + 28, 99_999
    ch_names = ["MISC", "STI101"] + ["c" + str(k) for k in range(n_channels - 2)]
    # NOTE: actually MISC and STI101 should have specific ch_types, but nevermind for now
    ch_names = ch_names[:n_channels]  # rectification
    content = np.random.randn(len(ch_names), n_times)
    content[:2, :] = 0  # non meg channels  # TODO what was that again?
    sfreq = schoffelen2019.RAW_SAMPLE_RATE
    for time_s, value in [(10, 10), (20, 20), (30, 10)]:
        t_ms = int(time_s * sfreq)
        content[1, t_ms: t_ms + 3000] = value  # adds events to the stimuli channel
    return mne.io.RawArray(content, info=mne.create_info(ch_names, sfreq=sfreq, ch_types='mag'))


def make_fake_events(total_duration: float = 83, seed: int = 1234) -> pd.DataFrame:
    """Create a fake event DataFrame with multiple events and precomputed blocks.
    """
    rng = random.Random(seed)
    event_dicts = list()
    wavpath = Path(__file__).parent.parent / 'mockdata' / 'one_two.wav'
    word_sequence = ['Toen', 'barkeeper', 'de']
    language = 'nl'

    time = 0.0
    for block_index in itertools.count():
        time += rng.uniform(0.5, 1.0)
        block_start_time = time

        # Need blocks to be approximately 3.0 s or more, otherwise they can't be used
        n_repeats = rng.randint(2, 3)
        sequence = word_sequence * n_repeats
        for word_index, word in enumerate(sequence):
            duration = rng.uniform(0.1, 0.2)
            time += duration + rng.uniform(0.1, 0.3)
            modality = rng.choice(['audio', 'visual'])

            # Add a word
            word_event = dict(kind='word', start=time, duration=duration, modality=modality,
                              language=language, word=word, word_index=word_index,
                              word_sequence=' '.join(sequence), condition='sentence')
            event_dicts.append(word_event)

            # Add a phoneme
            if modality == 'audio':
                ph_id = rng.choice(list(ph_dict.values()))
                phoneme_event = dict(kind='phoneme', start=time, duration=duration,
                                     phoneme_id=ph_id, modality=modality, language=language)
                event_dicts.append(phoneme_event)

        # Create corresponding sound event and block that cover the last events
        block_end_time = time + duration
        sound = dict(kind='sound', start=block_start_time,
                     duration=block_end_time - block_start_time, filepath=wavpath)
        event_dicts.append(sound)
        block = dict(kind='block', start=block_start_time,
                     duration=block_end_time - block_start_time, uid='block' + str(block_index))
        event_dicts.append(block)

        if time > total_duration:
            break

    events = pd.DataFrame(event_dicts).event.validate()
    return events

