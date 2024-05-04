# -*- coding: utf-8 -*-
"""<Explain your Codes>"""
# Ref from BrainMagic Project.
import mne
import numpy as np
import pandas as pd
from typing import List, Optional
import numpy as np
from numba import jit


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





def add_noise_to_data_with_snr(raw: mne.io.Raw, target_snr: float, noise_channel_proportion: float = None,
                               noise_types: List[str] = ['gaussian'], impulse_ratio: float = None,
                               noise_levels: Optional[List[float]] = None, sine_frequency: Optional[float] = None,
                               sine_amplitude: Optional[float] = None, high_amplitude_ratio: Optional[float] = None,
                               high_amplitude_duration: float = None,
                               high_frequency_noise_level: Optional[float] = None) -> np.ndarray:
    """
    Add noise to multi-channel time series data to achieve target signal-to-noise ratio (SNR).

    Parameters:
        data (mne.io.Raw): Input multi-channel time series data of shape (num_channels, num_samples).
        target_snr (float): Target signal-to-noise ratio (in dB) to achieve.
        noise_channel_proportion (float): Proportion of channels to add noise to (e.g., 0.5 for half of the channels). default:None,select all channels.
        noise_types (list): List of noise types to add. Options: 'gaussian', 'uniform', 'impulse', 'poisson',
                            'exponential', 'sine', 'high_amplitude', 'high_frequency'.
        noise_levels (list): List of noise levels corresponding to each noise type. If None, the noise levels will be
                              determined automatically to achieve the target SNR.
        impulse_ratio (float): Ratio of impulses in the noise.
        sine_frequency (float): Frequency of the sine wave to be added.
        sine_amplitude (float): Amplitude of the sine wave to be added.
        high_amplitude_ratio(float): Ratio of samples to be replaced by high amplitude noise.(Level of high amplitude noise to add.)
        high_amplitude_duration(int): Duration of the high amplitude noise (number of consecutive samples).
        high_frequency_noise_level (float): Level of high frequency noise to add.

    Returns:
        noisy_data (numpy.ndarray): Time series data with added noise to achieve the target SNR.
    """
    data = raw.get_data()
    num_channels, num_samples = data.shape

    if noise_levels is None:
        # Automatically determine noise levels to achieve the target SNR
        signal_power = np.sum(data ** 2) / (num_channels * num_samples)
        noise_power = signal_power / (10 ** (target_snr / 10))
        noise_levels = [np.sqrt(noise_power)] * len(noise_types)
        print("noise_levels:", noise_levels)
    if noise_channel_proportion is not None:
        num_noisy_channels = int(noise_channel_proportion * num_channels)
        noisy_channels = np.random.choice(num_channels, size=num_noisy_channels, replace=False)
        print(f"contaminated channels:{num_noisy_channels}...")
        if num_noisy_channels == 0:
            return
    else:
        num_noisy_channels = num_channels

    total_noise = np.zeros_like(data)

    for noise_type, noise_level in zip(noise_types, noise_levels):
        if noise_type == 'gaussian':
            print("add gaussian noise...")
            noise = np.random.normal(0, 1, size=(num_noisy_channels, num_samples)) * noise_level
        elif noise_type == 'uniform':
            print("add white noise noise...")
            noise = np.random.uniform(-1, 1, size=(num_noisy_channels, num_samples)) * noise_level
        elif noise_type == 'poisson':
            noise = (np.random.poisson(1, size=(num_noisy_channels, num_samples)) - 1) * noise_level
        elif noise_type == 'exponential':
            noise = (np.random.exponential(scale=1, size=(num_noisy_channels, num_samples)) - 1) * noise_level

        elif noise_type == 'sine' and noise_channel_proportion is not None:
            time = np.arange(num_samples)
            noise = np.sin(2 * np.pi * sine_frequency * time / num_samples)
            # Adjust amplitude of sine noise to match target SNR
            noise_scaling_factor = np.sqrt(noise_power / np.mean(noise ** 2))
            scaled_sine_noise = noise_scaling_factor * noise
            noise = np.tile(scaled_sine_noise, (num_noisy_channels, 1))
        elif noise_type == 'impulse' and noise_channel_proportion is not None:
            # Determine number of impulses
            num_impulses = int(num_samples * impulse_ratio)
            single_ch_num_impulses = int(num_impulses / num_channels)
            print("single_ch_num_impulses:", single_ch_num_impulses)

            # Generate impulse noise
            impulse_indices = np.random.choice(num_samples, size=single_ch_num_impulses, replace=False)
            print("impulse_indices:", impulse_indices)
            impulse_noise = np.zeros((1, num_samples))
            print("impulse_noise shape:", impulse_noise.shape)
            impulse_noise[:, impulse_indices] = np.random.normal(0, 1, size=single_ch_num_impulses) * (noise_level)
            noise = np.tile(impulse_noise, (num_noisy_channels, 1))

            # shuffle
            for i in range(noise.shape[0]):
                np.random.shuffle(noise[i])

            # # scale noise to match target SNR.
            # noise_scaling_factor = np.sqrt(noise_power / np.mean(noise ** 2))
            # noise = noise_scaling_factor * noise

        elif noise_type == 'high_amplitude' and noise_channel_proportion is not None:
            num_high_amp = int(num_samples * high_amplitude_ratio)
            single_ch_num_high_amp = int(num_high_amp / num_channels)
            print("single_ch_num_high_amp:", single_ch_num_high_amp)
            noise = np.zeros_like(data)
            for idx, select_noisy_ch in enumerate(noisy_channels):
                # Generate high amplitude noise
                high_amp_indices = np.random.choice(num_samples - high_amplitude_duration + 1,
                                                    size=single_ch_num_high_amp, replace=False)
                high_amp_noise = np.zeros((1, num_samples))

                # Generate high amplitude noise
                for idx, start_idx in enumerate(high_amp_indices):
                    high_amp_noise[:, start_idx:start_idx + high_amplitude_duration] = np.random.normal(0, 1,
                                                                                                        size=high_amplitude_duration)
                noise[select_noisy_ch] = high_amp_noise

        elif noise_type == 'high_frequency' and noise_channel_proportion is not None:
            high_frequency_noise = np.sin(2 * np.pi * high_frequency_noise_level * np.arange(num_samples) / num_samples)
            # Adjust amplitude of high frequency noise to match target SNR
            # noise_scaling_factor = np.sqrt(noise_power / np.mean(high_frequency_noise ** 2))
            # scaled_high_frequency_noise = noise_scaling_factor * high_frequency_noise
            # noise = scaled_high_frequency_noise
            noise = high_frequency_noise

        else:
            raise ValueError("Unsupported noise type.")

        if noise_channel_proportion is not None:
            for j, channel in enumerate(noisy_channels):
                if len(noisy_channels) == 1 or noise_type in ['high_frequency']:
                    total_noise[channel] += noise
                else:
                    total_noise[channel] += noise[j]
        else:
            total_noise += noise

    print("total_noise:", total_noise, np.sum(total_noise), total_noise.shape)

    # scale noise to match target SNR.
    noise_scaling_factor = np.sqrt(noise_power / np.mean(total_noise ** 2))
    total_noise = noise_scaling_factor * total_noise

    if noise_type in ["sine"]:
        data[noisy_channels] = noise
        noisy_data = data
    # elif noise_type in ["high_frequency"]:
    #     # data[noisy_channels] += noise
    #     noisy_data = data + total_noise
    # elif noise_type in ["high_frequency","impulse"]:
    #     # scale noise to match target SNR.
    #     noise_scaling_factor = np.sqrt(noise_power / np.mean(total_noise ** 2))
    #     total_noise = noise_scaling_factor * total_noise
    #     noisy_data = data + total_noise
    else:
        noisy_data = data + total_noise

    # Calculate Signal-to-Noise Ratio (SNR) for the entire data
    signal_power = np.sum(data ** 2) / (data.shape[0] * num_samples)
    noise_power = np.sum(total_noise ** 2) / (data.shape[0] * num_samples)
    snr = 10 * np.log10(signal_power / noise_power)
    print("SNR:", snr)

    signal_power = np.mean(data ** 2)
    noise_power = np.mean(total_noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    print("SNR——mean:", snr)

    # package mne.io.Raw
    info = raw.info
    noisy_data = mne.io.RawArray(noisy_data, info)

    return noisy_data

if __name__ == "__main__":
    snr = 120
    noisy_data = add_noise_to_data_with_snr(opm_raw, target_snr=snr, noise_types=['high_amplitude'],
                                            high_amplitude_ratio=0.2, high_amplitude_duration=10,
                                            noise_channel_proportion=0.5)
    plt.close()
    print(f"SNR{snr}" * 100)
    noisy_data.plot(scalings=scaling, start=100)
    noisy_data.compute_psd(fmin=0, fmax=100).plot()
    plt.show()

    # snr = 5
    # noisy_data = add_noise_to_data_with_snr(sti_raw,target_snr=snr, noise_types=['impulse'],noise_channel_proportion=0.8,impulse_ratio=0.5)
    # plt.close()
    # print(f"SNR{snr}"*100)
    # noisy_data.plot()
    # noisy_data.compute_psd(fmin=0,fmax=100).plot()
    # plt.show()

    # sti_raw.plot()
    # sti_raw.compute_psd(fmin=0,fmax=100).plot()
    # plt.show()
    # 加不同snr-level的impulse脉冲噪音
    # for snr in [1,2,3,4,5,6,7,8,9,10,20,30,40,50]:


    # snr = 20
    # noisy_data = add_noise_to_data_with_snr(sti_raw,target_snr=snr, noise_types=['high_frequency'],high_frequency_noise_level=80,noise_channel_proportion=0.8)
    # plt.close()
    # print(f"SNR{snr}"*100)
    # noisy_data.plot()
    # noisy_data.compute_psd(fmin=0,fmax=100).plot()
    # plt.show()


    # snr = 50
    # noisy_data = add_noise_to_data_with_snr(sti_raw,target_snr=snr, noise_types=['sine'],sine_frequency=60,noise_channel_proportion=0.)
    # plt.close()
    # print(f"SNR{snr}"*100)
    # noisy_data.plot()
    # noisy_data.compute_psd(fmin=0,fmax=100).plot()
    # plt.show()


    # 加不同snr-level高斯噪音
    # for snr in [1,2,3,4,5,6,7,8,9,10,20,30,40,50]:
    #     noisy_data = add_noise_to_data_with_snr(sti_raw,target_snr=snr, noise_types=['gaussian'])
    #     plt.close()
    #     print(f"SNR{snr}"*100)
    #     noisy_data.plot()
    #     noisy_data.compute_psd(fmin=0,fmax=100).plot()
    #     plt.show()

    # 加白噪音
    # for snr in [1,2,3,4,5,6,7,8,9,10,20,30,40,50]:
    #     noisy_data = add_noise_to_data_with_snr(sti_raw,target_snr=snr, noise_types=['uniform'])
    #     plt.close()
    #     print(f"SNR{snr}"*100)
    #     noisy_data.plot()
    #     noisy_data.compute_psd(fmin=0,fmax=100).plot()
    #     plt.show()
    # -----------------------------------------------------------------------------------------------------
    # 加白噪音，固定信噪比，加噪音通道越来越多
    # for i in np.arange(0,1.1,0.1):
    #     print("-"*100)
    #     print("select ratio:",i)
    #     noisy_data = add_noise_to_data_with_snr(sti_raw,target_snr=10, noise_types=['uniform'], noise_channel_proportion=i)
    #     plt.close()
    #     print(f"SNR{snr}"*100)
    #     noisy_data.plot()
    #     noisy_data.compute_psd(fmin=0,fmax=100).plot()
    #     plt.show()
