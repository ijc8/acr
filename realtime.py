import sys

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd


buffer = np.zeros(2048)

alphabet = ''.join(chr(ord("A") + i) for i in range(26))

colors = np.array([
    [0.0, 0.5, 0.8],
    [1.0, 0.0, 0.0]
])

recording = False
recorded = []
target = None

def callback(indata, frames, time, status):
    global i, buffer
    buffer[:-len(indata)] = buffer[len(indata):]
    buffer[-len(indata):] = indata[:, 0]
    history[i] = 10 * np.log10((np.diff(buffer)**2).mean())
    # active[i] = history[i] > -70
    active[i] = recording
    if recording:
        recorded.append(history[i])
    i = (i + 1) % len(history)
    plot.set_facecolor(colors[active])
    plot.set_offsets(np.c_[indices, history])

def find_device(name):
    for i, device in enumerate(sd.query_devices()):
        if name in device["name"]:
            return i
    return -1


if len(sys.argv) > 1:
    name = ' '.join(sys.argv[1:])
    device = find_device(name)
    if device < 0:
        exit(f"Could not find device matching '{name}'")
else:
    print(sd.query_devices())
    device = int(input("Enter device index: "))


history = np.zeros(512)
active = np.zeros(512, dtype=int)
indices = np.arange(len(history))
i = 0

from dtw import dtw
from scipy import stats

def on_press(event):
    global target, recording, recorded
    if event.key.upper() in alphabet:
        print(f"Recording {event.key.upper()}")
        if recording:
            print(target, recorded)
            recorded = []
        else:
            recording = True
        target = event.key.upper()
    if event.key == ' ':
        recording = not recording
        if not recording:
            recorded = np.array(recorded)
            print(target, recorded)
            if target:
                templates[target] = recorded
            else:
                alignments = [
                    dtw(stats.zscore(recorded), stats.zscore(template), dist_method="euclidean", keep_internals=True)
                    for template in templates.values()
                ]
                best = min(range(len(alignments)), key=lambda i: alignments[i].normalizedDistance)
                print(alphabet[best])
                # alignments[best].plot(type="threeway")
            target = None
            recorded = []
        print("Recording", "on" if recording else "off")

# import dtw_matcher, evaluate
# letters, fs = evaluate.load_dataset()
# letters = letters[0, :, 0]
# templates = {letter: dtw_matcher.get_power(data) for data, letter in zip(alphabet, letters)}
templates = {}

fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', on_press)
fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
ax = fig.add_subplot(111)
ax.set_ylim(-80, 6)
plot = ax.scatter(indices, history, s=1)
fig.show()

with sd.InputStream(channels=1, device=device, callback=callback, blocksize=512) as stream:
    try:
        while stream.active:
            fig.canvas.draw()
            fig.canvas.flush_events()
    except KeyboardInterrupt:
        print("User interrupt.")
