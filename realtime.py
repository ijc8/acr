import sys

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd


buffer = np.zeros(2048)

colors = np.array([
    [0.0, 0.5, 0.8],
    [1.0, 0.0, 0.0]
])

def callback(indata, frames, time, status):
    global i, buffer
    buffer[:-len(indata)] = buffer[len(indata):]
    buffer[-len(indata):] = indata[:, 0]
    history[i] = 10 * np.log10((np.diff(buffer)**2).mean())
    active[i] = history[i] > -40
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

fig = plt.figure()
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
