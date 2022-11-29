import mne
import reader

path = reader.read_data(disk='E:', sub=1, formate='snirf')
print(path)

data = mne.io.read_raw_snirf(fname=path)
print(data)
