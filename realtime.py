import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

alphabet = ''.join(chr(ord("A") + i) for i in range(26))

colors = np.array([
    [0.0, 0.5, 0.8],
    [1.0, 0.0, 0.0]
])

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

buffer = np.zeros(1024)
history = np.zeros(512)
active = np.zeros(512, dtype=int)
indices = np.arange(len(history))
i = 0

from dtw import dtw
from scipy import stats

def safe_dtw(query, reference):
    print(len(query), len(reference))
    try:
        return dtw(query, reference).normalizedDistance
    except ValueError:
        return 1
    

class LetterMatcher:
    def __init__(self):
        self.templates = {}
        self.recording = False
        self.recorded = []
        self.target = None

    def run(self):
        fig, (ax, self.ax2) = plt.subplots(2, 1)
        fig.canvas.mpl_connect('key_press_event', self.on_press)
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        ax.set_ylim(-80, 6)
        # plot = ax.scatter(indices, history, s=1)
        self.plot = ax.plot(indices, history)[0]
        # rects = ax2.bar(np.arange(26), np.random.random(26))
        # rects = dict(zip(alphabet, rects))
        self.rects = {}
        self.ax2.set_xticks(np.arange(26))
        self.ax2.set_xticklabels(alphabet)
        self.ax2.set_ylim(0, 1)
        fig.show()

        with sd.InputStream(channels=1, device=device, callback=self.process_audio, blocksize=512) as stream:
            try:
                while stream.active:
                    fig.canvas.draw()
                    fig.canvas.flush_events()
            except KeyboardInterrupt:
                print("User interrupt.")

    def stop_recording(self):
        recorded = np.array(self.recorded)
        if self.target:
            self.templates[self.target] = recorded
            for rect in self.rects.values():
                rect.remove()
            self.rects = dict(zip(
                self.templates.keys(),
                self.ax2.bar(np.arange(len(self.templates)), np.zeros(len(self.templates)), color='blue')
            ))
        elif self.templates:
            alignments = {
                letter: dtw(stats.zscore(recorded), stats.zscore(template), dist_method="euclidean", keep_internals=True)
                for letter, template in self.templates.items()
            }
            # max_dist = max(alignment.normalizedDistance for alignment in alignments.values())
            for letter, alignment in alignments.items():
                self.rects[letter].set_height(1 - alignment.normalizedDistance)
            best = min(alignments.keys(), key=lambda i: alignments[i].normalizedDistance)
            print(best)
            for rect in self.rects.values():
                rect.set_color("blue")
            self.rects[best].set_color("red")
            # alignments[best].plot(type="threeway")
        self.target = None
        self.recorded = []
        self.recording = False
    
    def update_plot(self, recorded):
        alignments = {
            letter: safe_dtw(stats.zscore(recorded), stats.zscore(template))
            for letter, template in self.templates.items()
        }
        # max_dist = max(alignment.normalizedDistance for alignment in alignments.values())
        for letter, alignment in alignments.items():
            self.rects[letter].set_height(1 - alignment)
        best = min(alignments.keys(), key=lambda i: alignments[i])
        print(best)
        for rect in self.rects.values():
            rect.set_color("blue")
        self.rects[best].set_color("red")
        # alignments[best].plot(type="threeway")

    def process_audio(self, indata, frames, time, status):
        global i, buffer
        buffer[:-len(indata)] = buffer[len(indata):]
        buffer[-len(indata):] = indata[:, 0]
        # history[i] = 10 * np.log10((np.diff(buffer)**2).mean())
        history[i] = 10 * np.log10((buffer**2).mean())
        # active[i] = history[i] > -70
        # active[i] = recording
        if self.recording:
            self.recorded.append(history[i])
            if self.templates and self.target is None:
                self.update_plot(np.array(self.recorded))
        i = (i + 1) % len(history)
        # plot.set_facecolor(colors[active])
        # plot.set_offsets(np.c_[indices, history])
        self.plot.set_data(indices, history)

    def on_press(self, event):
        if event.key.upper() in alphabet:
            # Record letter
            print(f"Recording {event.key.upper()}")
            if self.recording:
                self.stop_recording()
            self.recording = True
            self.target = event.key.upper()
        elif event.key == ' ':
            # Start recording for identification, or stop recording
            if self.recording:
                self.stop_recording()
            else:
                self.recording = True
            print("Recording", "on" if self.recording else "off")
        elif event.key == '=':
            with open('realtime.pkl', 'wb') as f:
                pickle.dump(self.templates, f)

# import dtw_matcher, evaluate
# letters, fs = evaluate.load_dataset()
# letters = letters[0, :, 0]
# templates = {letter: dtw_matcher.get_power(data) for data, letter in zip(alphabet, letters)}

lm = LetterMatcher()
lm.run()
