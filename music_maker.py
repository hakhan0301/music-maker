from Data import midi_io
import matplotlib.pyplot as plt

piano_roll = midi_io.load_midi_as_piano_roll("PKMN_X_Y_Route_8_By_Incinium.mid")
plt.imshow(piano_roll[:200])
plt.show()