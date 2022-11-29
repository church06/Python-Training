import mne
import reader

path = reader.read_data(disk='E:', sub=1)
data = mne.io.read_raw_snirf()
