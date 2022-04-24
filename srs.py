"""
Calculate Shock Response Spectrum (SRS) from accelerometer data 
using Smallwood ramp invariant method (http://www.vibrationdata.com/ramp_invariant/DS_SRS1.pdf)

Author: dsholes
Date: November 8, 2018
Version: 0.1

Credits: 
- Tom Irvine wrote a similar python module, and I am borrowing the idea to use scipy.signal.lfilter
- David Smallwood developed the algorithm

"""

from scipy.signal import lfilter
from scipy import integrate
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Constants
G_TO_MPS2 = 9.81

# Plot formatting
COLORS = sns.color_palette().as_hex()
AX_LABEL_FONT_DICT = {'size': 14}
AX_TITLE_FONT_DICT = {'size': 16}


def read_table(path_to_tsv):
    """

    NOTE ON DATA FORMAT:
        - TSV must be 2 columns
            - Time (in sec) is first column
            - Accel (in G's) is second column
        - No column headers (i.e. first row should be data)

    """

    input_data = path_to_tsv
    cols = [0, 1]
    col_names = ['time_s', 'accel_g']
    input_df = pd.read_table(input_data,
                             sep=' ',
                             header=None,
                             names=col_names,
                             usecols=cols)
    return read_dataframe(input_df)


def read_tsv(path_to_tsv):
    """

    NOTE ON DATA FORMAT:
        - TSV must be 2 columns
            - Time (in sec) is first column
            - Accel (in G's) is second column
        - No column headers (i.e. first row should be data)

    """

    input_data = path_to_tsv
    cols = [0, 1]
    col_names = ['time_s', 'accel_g']
    input_df = pd.read_csv(input_data,
                           sep='\t',
                           header=None,
                           names=col_names,
                           usecols=cols)
    return read_dataframe(input_df)


def read_csv(path_to_csv):
    """

    NOTE ON DATA FORMAT:
        - CSV must be 2 columns
            - Time (in sec) is first column
            - Accel (in G's) is second column
        - No column headers (i.e. first row should be data)

    """

    input_data = path_to_csv
    cols = [0, 1]
    col_names = ['time_s', 'accel_g']
    input_df = pd.read_csv(input_data,
                           header=None,
                           names=col_names,
                           usecols=cols)
    return read_dataframe(input_df)


def read_dataframe(input_df):
    '''
    Takes pandas DataFrame object
        - Time (in sec) is first column
        - Accel (in G's) is second column
    '''
    try:
        input_df.apply(pd.to_numeric)
    except ValueError:
        raise Exception("Input data file should not have column headers")
    srs_object = ShockResponseSpectrum(input_df)

    return srs_object


def build_nat_freq_array(fn_start=10., fn_end=1.0e4, oct_step_size=(1./12.)):
    fn_array = [fn_start]
    for i in range(int(fn_end-fn_start)):
        new_fn = (fn_start*2.**(oct_step_size))
        fn_array.append(new_fn)
        fn_start = new_fn
        if fn_start > fn_end:
            break
    fn_array = np.array(fn_array)
    return fn_array


def build_nat_freq_array_step(fn_start=10., fn_end=1.0e4, num_freq=1000):
    fn_array = [fn_start]
    for i in range(int(fn_end-fn_start)):
        new_fn = (fn_start+(((fn_end+1)-fn_start)/num_freq))
        fn_array.append(new_fn)
        fn_start = new_fn
        if fn_start > fn_end:
            break
    fn_array = np.array(fn_array)
    return fn_array


def remove_sensor_bias(input_accel_g):
    input_accel_g = input_accel_g - input_accel_g.mean()
    return input_accel_g


class ShockResponseSpectrum:
    def __init__(self, input_df):
        self.input_df = input_df

        self.input_time_s = self.input_df.time_s.values
        self.input_accel_g = self.input_df.accel_g.values

        # convert accel to m/s^2 for integration to velocity (m/s)
        self.input_accel_mps2 = self.input_accel_g * G_TO_MPS2
        self.input_vel_mps = integrate.cumtrapz(
            self.input_accel_mps2, self.input_time_s, initial=0.)

    def run_srs_analysis(self, fn_array, Q=10, remove_bias=False):
        """
        Review Smallwood method in his paper:
            - 'AN IMPROVED RECURSIVE FORMULA FOR CALCULATING SHOCK RESPONSE SPECTRA'
            - http://www.vibrationdata.com/ramp_invariant/DS_SRS1.pdf
        """
        self.Q = Q
        self.fn_array = fn_array
        self.remove_bias = remove_bias

        if self.remove_bias:
            # Remove bias, talk to Martin about this
            self.input_accel_g = remove_sensor_bias(self.input_accel_g)
            # convert accel to m/s^2 for integration to velocity (m/s)
            self.input_accel_mps2 = self.input_accel_g * G_TO_MPS2
            self.input_vel_mps = integrate.cumtrapz(
                self.input_accel_mps2, self.input_time_s, initial=0.)
            print('Input data has been modified to remove sensor bias (offset)...')

        # Should I give user access to the following coefficients??
        damp = 1./(2.*self.Q)
        T = np.diff(self.input_time_s).mean()  # sample interval
        omega_n = 2. * np.pi * self.fn_array
        omega_d = omega_n * np.sqrt(1 - damp**2.)
        E = np.exp(-damp * omega_n * T)
        K = T*omega_d
        C = E*np.cos(K)
        S = E*np.sin(K)
        S_prime = S/K
        b0 = 1. - S_prime
        b1 = 2. * (S_prime - C)
        b2 = E**2. - S_prime
        # Necessary because of how scipy.signal.lfilter() is structured
        a0 = np.ones_like(self.fn_array)
        a1 = -2. * C
        a2 = E**2.
        b = np.array([b0, b1, b2]).T
        a = np.array([a0, a1, a2]).T

        # Calculate SRS using Smallwood ramp invariant method
        self.pos_accel = np.zeros_like(self.fn_array)
        self.neg_accel = np.zeros_like(self.fn_array)
        self.abs_accel = np.zeros_like(self.fn_array)
        for i, f_n in enumerate(self.fn_array):
            output_accel_g = lfilter(b[i], a[i], self.input_accel_g)
            self.pos_accel[i] = output_accel_g.max()
            self.neg_accel[i] = np.abs(output_accel_g.min())
            self.abs_accel[i] = max(
                output_accel_g.max(), np.abs(output_accel_g.min()))

    def _make_srs_subplot(self, ax, requirement, type="absmax"):

        if type == "max":
            ax.loglog(self.fn_array, self.pos_accel,
                      label='Positive',
                      color=COLORS[0],
                      linestyle='dashdot')
        elif type == "min":
            ax.loglog(self.fn_array, self.neg_accel,
                      label='Negative',
                      color=COLORS[1],
                      linestyle='--')
        elif type == "absmax":
            ax.loglog(self.fn_array, self.abs_accel,
                      label='Abs Max',
                      color=COLORS[2],
                      linestyle='-')
        elif type == "overplot":
            ax.loglog(self.fn_array, self.pos_accel,
                      label='Positive',
                      color=COLORS[0],
                      linestyle='dashdot')
            ax.loglog(self.fn_array, self.neg_accel,
                      label='Negative',
                      color=COLORS[1],
                      linestyle='--')
            ax.loglog(self.fn_array, self.abs_accel,
                      label='Abs Max',
                      color=COLORS[2],
                      linestyle='-')

        if requirement is not None:
            self.protocol_fn, self.protocol_accel = requirement
            ax.loglog(self.protocol_fn, self.protocol_accel,
                      color=COLORS[3],
                      linewidth=2,
                      label='Requirement')

        leg = ax.legend(fancybox=True,
                        framealpha=1,
                        frameon=True)
        leg.get_frame().set_edgecolor('k')
        ax.grid(True, which="both")
        ax.set_xlabel('Natural Frequency (Hz)', fontdict=AX_LABEL_FONT_DICT)
        ax.set_ylabel('Peak Accel (G)', fontdict=AX_LABEL_FONT_DICT)
        ax.set_title('Acceleration Shock Response Spectrum (Q={0:.1f})'.format(self.Q),
                     fontdict=AX_TITLE_FONT_DICT)
        return ax

