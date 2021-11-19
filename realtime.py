import sys

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd


def callback(indata, frames, time, status):
    global i
    indata = indata[:, 0]
    history[i] = 10 * np.log10((np.diff(indata)**2).sum())
    i = (i + 1) % len(history)
    line.set_ydata(history)

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
i = 0

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim(-60, 6)
line = ax.plot(history)[0]
fig.show()

with sd.InputStream(channels=1, device=device, callback=callback, blocksize=512) as stream:
    try:
        while stream.active:
            fig.canvas.draw()
            fig.canvas.flush_events()
    except KeyboardInterrupt:
        print("User interrupt.")
