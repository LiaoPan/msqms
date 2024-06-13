# -*- coding: utf-8 -*-
"""Visual Inspection"""
import mne
import seaborn
import plotly
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import plotly.graph_objs as go
import plotly.offline as pyo

class VisualInspection(object):
    def __init__(self, raw):
        self.raw = raw

    def _downsample_mask(self, mask, downsample_dim=1000):
        """Downsample the heatmap to a specific dimension to handle excessive time.

        The heatmap dimensions displayed by default are set according to `downsample_dim`, that is, no matter how long the data is,
        it is compressed to `downsample_dim` and its data is summed.
        The advantage of this method is that it can cover no matter how long it is. [Default Recommendation]

        Parameters
        ----------
        mask : numpy.ndarray[bool]
            A boolean mask with the same shape as the data.
        downsample_dim : int
            The dimension value to which mask is reduced.
        Returns
        -------

        """
        n_row = mask.shape[0]
        n_col = mask.shape[1]
        interval_size = np.ceil(n_col / downsample_dim).astype(int)
        down_mask = np.zeros((n_row, downsample_dim), dtype=int)
        for i in range(downsample_dim):
            down_mask[:, i] = np.sum(mask[:, i * interval_size:(i + 1) * interval_size], axis=1)
        return down_mask

    def visualize_heatmap(self,data, bad_mask, sfreq=1000, label='NaN', adaptive=True, downsample_dim=1000):
        """Visualize the positions of NaN values in multi-channel brain data matrix and display the percentage of NaN values. [Implemented based on plotly]

            Parameters
            ----------
            data : numpy.ndarray
                Multi-channel brain data matrix.
            bad_mask : numpy.ndarray
                matrix containing indices of bad values(NaN\Bad Segments\Zeros\constant value etc).
            sfreq : float
                sample frequency in Hz
            label :
                the label of heatmap
            adaptive : bool
                deal with the long time problem when plotting the heatmap.
            downsample_dim : int
                The heatmap dimensions displayed by default are set according to downsample_dim,
                that is, no matter how long the data is, it is compressed to downsample_dim and its data is summed.
            title : str
                the title of heatmap.

            Returns
            -------
                None
        """

        # Calculate bad percentage
        total_samples = data.shape[0] * data.shape[1]
        bad_percentage = np.sum(bad_mask) / total_samples * 100

        # split the mask
        if adaptive:
            bad_mask = self._downsample_mask(bad_mask, downsample_dim)

        # Create a figure and plot the NaN mask
        # plt.figure(figsize=(24, 6))
        # plt.imshow(bad_mask, aspect='auto', cmap='YlGnBu', interpolation='none') # binary coolwarm

        godata = [
            go.Heatmap(
                z=bad_mask,
                colorscale='Viridis'
            )
        ]

        # customize xlabel
        interval_time = data.shape[1] / (sfreq * downsample_dim)
        xlabels = [f"{interval_time * i}s" for i in np.arange(0, downsample_dim, int(downsample_dim / 10))]
        # plt.xticks(np.arange(0,downsample_dim,int(downsample_dim/10)),xlabels)

        # create layout
        layout = go.Layout(
            title=f'{label} Values in MEG Data, {label} Percentage: {bad_percentage:.6f}%',
            # xaxis=dict(title='Time'),
            yaxis=dict(title='Channel'),
            xaxis=dict(
                title='Time',
                tickvals=list(np.arange(0, downsample_dim, int(downsample_dim / 10))),
                ticktext=xlabels,
            ),
            title_x=0.5,
            width=1200,
            height=800,

        )

        # draw a diagram
        fig = go.Figure(data=godata, layout=layout)
        fig.show()

    def visualize_bad_segments(data, bad_segment_indices):
        """
        Visualize the positions of bad segments in multi-channel brain data along with the percentage of bad segments.

        Parameters:
            data (numpy.ndarray): Multi-channel brain data.
            bad_segment_indices (list): List of tuples indicating start and end indices of bad segments for each channel.

        Returns:
            None
        """
        num_channels = data.shape[1]

        # Create figure and axis
        fig, axs = plt.subplots(num_channels, 1, figsize=(10, 2 * num_channels), subplot_kw={'projection': 'polar'})

        for i, ax in enumerate(axs):
            # Generate data for spiral plot
            theta = np.linspace(0, 10 * np.pi, data.shape[0])
            r = data[:, i]

            # Plot the spiral
            ax.plot(theta, r)

            # Highlight bad segments
            for start, end in bad_segment_indices[i]:
                ax.fill_between(theta[start:end], r[start:end], color='red', alpha=0.3)

            # Set title for each subplot
            ax.set_title(f'Channel {i + 1}')

        # Adjust layout
        plt.tight_layout()

        # Show the plot
        plt.show()

    def visualize_nan_values(data, nan_mask):
        """
        Visualize the positions of NaN values in multi-channel brain data matrix and display the percentage of NaN values.

        Parameters:
            data (numpy.ndarray): Multi-channel brain data matrix.
            nan_mask (numpy.ndarray): matrix containing indices of NaN values.

        Returns:
            None
        """

        # Create a figure and plot the NaN mask
        plt.figure(figsize=(10, 6))
        plt.imshow(nan_mask, aspect='auto', cmap='YlGnBu', interpolation='none')  # binary coolwarm

        # Calculate NaN percentage
        total_samples = data.shape[0] * data.shape[1]
        nan_percentage = np.sum(nan_mask) / total_samples * 100

        # Add title and labels
        plt.title(f'NaN Values in MEG Data, NaN Percentage: {nan_percentage:.2f}%')
        plt.ylabel('Channel')
        plt.xlabel('Time')

        # Show colorbar
        plt.colorbar(label='NaN')

        # Show the plot
        plt.show()

    def visualize_bad_segments(data, bad_segment_mask):
        """
        Visualize the positions of bad segments in multi-channel brain data matrix and display the percentage of bad segments.

        Parameters:
            data (numpy.ndarray): Multi-channel brain data matrix.
            bad_segment_mask (numpy.ndarray): 2D binary mask indicating the positions of bad segments.

        Returns:
            None
        """
        # Create a figure and plot the bad segment mask
        plt.figure(figsize=(10, 6))
        plt.imshow(bad_segment_mask, aspect='auto', cmap='YlGnBu', interpolation='none')  # Reds

        # Calculate bad segment percentage
        total_samples = data.shape[0] * data.shape[1]
        bad_segment_percentage = np.sum(bad_segment_mask) / total_samples * 100

        # Add title and labels
        plt.title(f'Bad Segments in Multi-channel Brain Data, Bad Segment Percentage: {bad_segment_percentage:.2f}%')
        plt.ylabel('Channel')
        plt.xlabel('Time')

        # Show colorbar
        plt.colorbar(label='Bad Segment')

        # Show the plot
        plt.show()

    def visualize_zero_values(data, zero_mask):
        """
        Visualize the positions of zero values in multi-channel brain data matrix and display the percentage of zero values.

        Parameters:
            data (numpy.ndarray): Multi-channel brain data matrix.
            zero_mask (numpy.ndarray): Matrix containing indices of zero values.

        Returns:
            None
        """

        # Create a figure and plot the zero mask
        plt.figure(figsize=(10, 6))
        plt.imshow(zero_mask, aspect='auto', cmap='YlGnBu', interpolation='none')  # binary

        # Calculate zero percentage
        total_samples = data.shape[0] * data.shape[1]
        zero_percentage = np.sum(zero_mask) / total_samples * 100

        # Add title and labels
        plt.title(f'Zero Values in MEG Data, Zero Percentage: {zero_percentage:.2f}%')
        plt.xlabel('Time')
        plt.ylabel('Channel')

        # Show colorbar
        plt.colorbar(label='Zero')

        # Show the plot
        plt.show()

    def visual_psd(raw):
        from mne.viz._mpl_figure import _line_figure, _split_picks_by_type
        from mne.defaults import _handle_default
        try:
            from mne._fiff.pick import _picks_to_idx
        except ImportError as e:
            print(e)

        scalings = _handle_default("scalings", None)
        units = _handle_default("units", None)
        titles = _handle_default("titles", None)

        # split picks by channel type
        picks = _picks_to_idx(
            raw.info, None, "data", exclude=(), with_ref_meg=False
        )
        (picks_list, units_list, scalings_list, titles_list) = _split_picks_by_type(
            raw, picks, units, scalings, titles
        )

        for idx, pi in enumerate(picks_list):
            pick_raw = raw.copy().pick(pi)
            psd = pick_raw.compute_psd(verbose=False)
            df = psd.to_data_frame()
            scaling = scalings_list[idx]
            df_log = df.drop(columns=['freq']).apply(
                lambda x: 10 * np.log10(np.maximum(x * scaling ** 2, np.finfo(float).tiny)), axis=0)

            # 将对数转换后的结果与 'freq' 列合并
            # Merge the log-converted result with the 'freq' column
            df_log['freq'] = df['freq']

            # 创建频谱图的数据
            df = df_log
            traces = []
            for column in df.columns[:-1]:
                trace = go.Scatter(
                    x=df['freq'],
                    y=df[column],
                    mode='lines',
                    name=column
                )
                traces.append(trace)

            unit = units_list[idx]
            if "/" in unit:
                unit = f"({unit})"
            ylabel = f'{unit}²/Hz (dB)'  # "fT²/Hz (dB)"
            layout = go.Layout(
                title=titles_list[idx],
                xaxis=dict(title='Frequency (Hz)'),
                yaxis=dict(title=ylabel),
                width=1200,
                height=800,
                title_x=0.5,  # center title
            )

            fig = go.Figure(data=traces, layout=layout)
            fig.show()
            pyo.plot(fig, filename=f'spectrum_plot_opm_visual_{titles_list[idx]}.html')

    def plot_multivariate_time_series(data):
        """
        Plot the mean, standard deviation, and variance of a multivariate time series using barplot.

        Parameters:
        data (ndarray): Multivariate time series data with shape (samples, channels).

        Returns:
        None, directly plots the barplot.
        """
        # Calculate mean, standard deviation, and variance
        mean_values = np.mean(data, axis=1)
        std_values = np.std(data, axis=1)
        var_values = np.var(data, axis=1)

        # Convert the results to DataFrame format
        result_df = pd.DataFrame({'Channel': np.arange(1, data.shape[0] + 1),
                                  'Mean': mean_values,
                                  'Std': std_values,
                                  'Variance': var_values})
        # Plot the barplot
        fig, ax = plt.subplots(1, 2, figsize=(12, 24))
        plt.tight_layout()
        sns.barplot(data=result_df, x='Mean', y='Channel', color='#D5A19C', label='Mean', orient='h',
                    ax=ax[0])  # color='skyblue',
        sns.barplot(data=result_df, y='Channel', x='Std', label='Std', color='#A0BADB', orient='h',
                    ax=ax[0])  # color='orange',
        sns.barplot(data=result_df, y='Channel', x='Variance', label='Variance', color='#A4CBCC', orient='h',
                    ax=ax[1])  # color='green',

        ax[0].set_title('Mean, Standard Deviation of Each Channel')
        ax[1].set_title('Variance of Each Channel')
        ax[0].legend()
        ax[1].legend()

        # Add labels and title
        ax[0].set_ylabel('Channel')
        ax[0].set_xlabel('Value')
        ax[1].set_ylabel('Channel')
        ax[1].set_xlabel('Value')
        # plt.suptitle('Mean, Standard Deviation, and Variance of Each Channel')
        # plt.title()
        # plt.legend()

        plt.show()

    def visual_bad_channel_topomap(self, bad_channels: list, show_names: bool = False):
        """
        plot bad channel topomap
        Parameters
        ----------
        raw(MNE.Raw): mne raw object
        bad_channels(list): the names of bad channels
        show_names(bool): whether to display all channel names.

        Returns
            instance of Figure
            Figure containing the sensor topography.
        -------

        """
        raw = self.raw.copy()
        raw.info['bads'] = bad_channels
        fig = raw.plot_sensors(show_names=show_names)
        return fig

    def visual_bad_channels_distribution(mask, ch_names, mode, fontsize=10):
        sns.set(style="white")

        if mode == 'squid':
            plt.figure(figsize=(5, 12))
            orient = 'h'
            fontsize = 0.8 * fontsize
        else:
            plt.figure(figsize=(24, 6))
            orient = 'v'

        ax = sns.barplot(data=mask, orient=orient)
        # add text labels with each bar's value.
        labels = []
        for idx, m in enumerate(mask.to_numpy()[0]):
            if m == 1:
                labels.append(ch_names[idx])
            else:
                labels.append('')

        # get ratio of bad channels
        # ratio = mask.iloc[0].sum()/mask.size
        if mode == 'squid':
            plt.xticks([])
            plt.yticks([0, len(labels)], [0, len(labels)])
            plt.ylabel('Channel Index')
            plt.xlabel('Bad Channels')
        else:
            plt.xticks([0, len(labels)], [0, len(labels)])
            plt.yticks([])
            plt.xlabel('Channel Index')
            plt.ylabel('Bad Channels')

        ax.bar_label(ax.containers[0], labels=labels, fontsize=fontsize)
        #

        plt.title('Bad Channels Distribution')

    def visual_bad_channels_distribution2(mask, ch_names, mode, fontsize=8):
        sns.set(style="white")
        plt.figure(figsize=(24, 6))
        ax = sns.barplot(data=mask)
        # add text labels with each bar's value.
        labels = []
        for idx, m in enumerate(mask.to_numpy()[0]):
            if m == 1:
                labels.append(ch_names[idx])
            else:
                labels.append('')

        # get ratio of bad channels
        ratio = mask.iloc[0].sum() / mask.size
        if ratio >= 0.1 and mode == 'squid':
            plt.xticks([0, len(labels)], [0, len(labels)])
        else:
            ax.bar_label(ax.containers[0], labels=labels, fontsize=fontsize)
            plt.xticks([])
        plt.yticks([])
        plt.xlabel('Channel Index')
        plt.ylabel('Bad Channels')
        plt.title('Bad Channels Distribution')

    def plot_average_psd(self):
        """
        Power Spectral Density average on time.
        Returns
        -------

        """
        pass

    def plot_power_on_ts(self):
        pass

    def plot_chan_variance_ts(self):
        """
        channel variance time series.
        Returns
        -------

        """

    def plot_average_freq(self):
        """
        Returns
        -------

        """
        pass

    def plot_constant_dist(self):
        """
        constant value time series.
        Returns
        -------

        """

    def plot_bad_channel_topo(self):
        """
        The bad channels topomap.
        Returns
        -------
        """
        pass

    def plot_bad_channel_dist(self):
        """
        The bad channels distribution.
        Returns
        -------
        """
        pass

    def plot_bad_segment_dist(self):
        pass

    def plot_bad_trails_dist(self):
        pass

    def plot_no_signals(self):
        pass

    def plot_high_amplitude(self):
        pass

    def plot_low_freq_and_high_amp(self):
        """
        Low frequency and high amplitude.
        Returns
        -------
        """
        pass



    def visualize_heatmap(self,data, bad_mask, sfreq=1000, label='NaN', adaptive=True, downsample_dim=1000):
        """Visualize the positions of NaN values in multi-channel brain data matrix and display the percentage of NaN values.
        
        Parameters:
            data : numpy.ndarray
                Multi-channel brain data matrix.
            bad_mask : numpy.ndarray
                matrix containing indices of bad values(NaN\Bad Segments\Zeros\constant value etc).
            adaptive : bool
                deal with the long time problem when plotting the heatmap.
            adaptive_n : int
                divide all time points into N buckets.
            title : str
                the title of heatmap.
        Returns:
            None
        """
        # Calculate bad percentage
        total_samples = data.shape[0] * data.shape[1]
        bad_percentage = np.sum(bad_mask) / total_samples * 100

        # split the mask
        bad_mask = self._downsample_mask(bad_mask, downsample_dim)

        # Create a figure and plot the NaN mask
        plt.figure(figsize=(24, 6))
        plt.imshow(bad_mask, aspect='auto', cmap='YlGnBu', interpolation='none')  # binary coolwarm

        # Add title and labels
        plt.title(f'{label} Values in MEG Data, {label} Percentage: {bad_percentage:.6f}%')
        plt.ylabel('Channel')
        plt.xlabel('Time')

        # Show colorbar
        plt.colorbar(label=label)

        # customize xlabel
        interval_time = data.shape[1] / (sfreq * downsample_dim)
        xlabels = [f"{interval_time * i}s" for i in np.arange(0, downsample_dim, int(downsample_dim / 10))]
        plt.xticks(np.arange(0, downsample_dim, int(downsample_dim / 10)), xlabels)

        # Show the plot
        plt.show()

if __name__ == '__main__':
    import mne
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    opm_mag_fif = r"C:\Data\Datasets\OPM-Artifacts\S01.LP.fif"
    opm_raw = mne.io.read_raw(opm_mag_fif, verbose=False)
    opm_raw.first_samp

    # squid_fif = Path(r"C:\Data\Datasets\MEG_Lab\02_liaopan\231123\run1_tsss.fif")
    # squid_raw = mne.io.read_raw_fif(squid_fif)
    # print(squid_raw.first_samp)
    # print(squid_raw.first_time)
    # squid_raw.time_as_index(7000)
    # squid_data = squid_raw.get_data('mag', start=0, stop=6000)
    # print("squid_data shape:", squid_data.shape)

    opm_data = opm_raw.get_data('mag', start=0, stop=6000)
    print("opm_data shape:", opm_data.shape)

    opm_mag_visual = r"C:\Data\Datasets\全记录数据\opm_visual.fif"
    opm_raw_visual = mne.io.read_raw(opm_mag_visual, verbose=False)
    opm_data_visual = opm_raw_visual.get_data('mag', start=0, stop=200)
    opm_data_visual_2 = opm_data_visual.get_data('mag')

    VisualInspection()


