import streamlit as st
import numpy as np
import srs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Title of the project

st.title("SHOCK RESPONSE SPECTRUM")

# Select the types of SRS plot to include

srs_type = st.sidebar.selectbox("Select the SRS Type", ["absmax", "max", "min", "overplot"], index=0)

# Ask user for the data file

input_data = st.file_uploader("Please add the time history data file: ")

# Read data file, time (s) first column, accel (G) second column
# NO COLUMN HEADERS!!!
# A ShockResponseSpectrum object is created
# Select the type of delimiter in the time history file

delim = st.sidebar.radio("FILE DELIMITER", ["tab", "space", "comma"], index=1)

# Build array of natural frequencies for SRS
# similar to np.logspace, but using octaves as the step size
# User input natural frequency array build

min_freq = st.sidebar.text_input("Starting Frequency (Hz)", "1")
max_freq = st.sidebar.text_input("Ending Frequency (Hz)", "5000")


# Building natural frequency spacing options

freq_spacing = st.sidebar.selectbox("Frequency Spacing: ", ["Linear", "1/N Octave"], index=0)
q_factor = st.sidebar.text_input("Q Factor: ", "10")

if freq_spacing == "1/N Octave":
    freq_p_octave = float(st.sidebar.text_input("Freq/Octave"))
    fn_test_array = srs.build_nat_freq_array(fn_start = int(min_freq), fn_end = int(max_freq), oct_step_size = freq_p_octave)
elif freq_spacing == "Linear":
    total_frequencies = int(st.sidebar.text_input("Number of Frequencies", "1000"))
    fn_test_array = srs.build_nat_freq_array_step(fn_start = int(min_freq), fn_end = int(max_freq), num_freq = total_frequencies)

if input_data:
    if delim == "tab":
        srs_test_object = srs.read_tsv(input_data)
    elif delim == "space":
        srs_test_object = srs.read_table(input_data)
    elif delim == "comma":
        srs_test_object = srs.read_csv(input_data)

    srs_test_object.run_srs_analysis(fn_test_array, Q = int(q_factor))


# Plot results. SRS Requirements from test protocol can be added to plot. Following snippet is spec for PSLV project
    protocol_fn = np.array([100.,1000.,5000.])
    protocol_accel = np.array([30.,1500.,1500.])
    test_reqs = (protocol_fn, protocol_accel)

    fig = plt.figure()

    gs = gridspec.GridSpec(3, 4)
    gs.update(hspace=0.5, wspace=0.75)


    ax = fig.add_subplot(gs[:, :])

    ax = srs_test_object._make_srs_subplot(ax, requirement=test_reqs, type=srs_type)

    fig.set_size_inches(10, 10)
    st.pyplot(fig)
    fig.savefig("shockresponse.pdf", dpi=600)
