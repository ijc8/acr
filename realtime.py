import pickle
import sys

import matplotlib.pyplot as plt
from matplotlib import collections as mc
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
spectrogram = np.zeros((512, len(buffer)//2+1))
active = np.zeros(512, dtype=int)
indices = np.arange(len(history))
i = 0

from dtw import dtw
from scipy import stats

def safe_dtw(query, reference):
    try:
        return dtw(query, reference).normalizedDistance
    except ValueError:
        return 1

from words import match_sequence

class LetterMatcher:
    def __init__(self):
        with open("realtime.pkl", "rb") as f:
            self.templates = pickle.load(f)
        self.recording = False
        self.recorded = []
        self.target = None

    def run(self):
        fig, ((ax, self.ax4), (self.ax2, self.ax3)) = plt.subplots(2, 2)
        fig.canvas.mpl_connect('key_press_event', self.on_press)
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        ax.set_ylim(-80, 6)
        # plot = ax.scatter(indices, history, s=1)
        self.plot = ax.plot(indices, history)[0]
        # rects = ax2.bar(np.arange(26), np.random.random(26))
        # rects = dict(zip(alphabet, rects))
        self.ax2.set_xticks(np.arange(26))
        self.ax2.set_xticklabels(alphabet)
        self.ax2.set_ylim(0, 1)
        self.rects = dict(zip(
            self.templates.keys(),
            self.ax2.bar(np.arange(len(self.templates)), np.zeros(len(self.templates)), color='blue')
        ))

        self.im = self.ax4.imshow(spectrogram.T, origin='lower', aspect='auto', interpolation='none', vmin=-60, vmax=0)
        fig.show()

        with sd.InputStream(channels=1, device=device, callback=self.process_audio, blocksize=512) as stream:
            try:
                while stream.active:
                    fig.canvas.draw()
                    fig.canvas.flush_events()
            except KeyboardInterrupt:
                print("User interrupt.")

    def stop_recording(self):
        self.recording = False
        # Lop off the keyboard sounds.
        recorded = np.array(self.recorded)[5:-5]
        if self.target == "multi":
            templates = list(self.templates.values())
            labels = list(self.templates.keys())
            predicted, signal, templates, path = match_sequence(templates, labels, recorded, return_internals=True)
            template_concat = np.concatenate(templates)
            print(predicted)
            self.ax3.clear()
            offset = 4
            dtwPlotTwoWay(self.ax3, signal, template_concat, path[:, 0], path[:, 1], offset=offset)
            x = len(templates[0])
            for template in templates[1:]:
                self.ax3.plot(np.arange(len(template)) + x, template + offset)
                x += len(template)
        elif self.target:
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
            for letter, alignment in alignments.items():
                self.rects[letter].set_height(1 - alignment.normalizedDistance)
            best = min(alignments.keys(), key=lambda i: alignments[i].normalizedDistance)
            print("Matched:", best)
            for rect in self.rects.values():
                rect.set_color("blue")
            self.rects[best].set_color("red")
            # alignments[best].plot(type="threeway", ax=self.ax3)
            self.ax3.clear()
            align = alignments[best]
            dtwPlotTwoWay(self.ax3, align.query, align.reference, align.index1, align.index2, offset=4)
        self.target = None
        self.recorded = []
    
    def update_plot(self, recorded):
        alignments = {
            letter: safe_dtw(stats.zscore(recorded), stats.zscore(template))
            for letter, template in self.templates.items()
        }
        for letter, alignment in alignments.items():
            self.rects[letter].set_height(1 - alignment)
        best = min(alignments.keys(), key=lambda i: alignments[i])
        for rect in self.rects.values():
            rect.set_color("blue")
        self.rects[best].set_color("red")

    def process_audio(self, indata, frames, time, status):
        global i, buffer
        buffer[:-len(indata)] = buffer[len(indata):]
        buffer[-len(indata):] = indata[:, 0]
        # history[i] = 10 * np.log10((np.diff(buffer)**2).mean())
        history[i] = 10 * np.log10((buffer**2).mean())
        spectrogram[i] = 20 * np.log10(np.abs(np.fft.rfft(buffer)))
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
        self.im.set_data(spectrogram.T)

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
        elif event.key == '.':
            if self.recording:
                self.stop_recording()
            else:
                self.recording = True
                self.target = "multi"
        elif event.key == '=':
            with open('realtime.pkl', 'wb') as f:
                pickle.dump(self.templates, f)

# import dtw_matcher, evaluate
# letters, fs = evaluate.load_dataset()
# letters = letters[0, :, 0]
# templates = {letter: dtw_matcher.get_power(data) for data, letter in zip(alphabet, letters)}


# NOTE: Adapted from https://github.com/DynamicTimeWarping/dtw-python/blob/master/dtw/dtwPlot.py,
# to support plotting to pre-existing axes.
def dtwPlotTwoWay(ax, xts, yts, index1, index2, offset=0):
    maxlen = max(len(xts), len(yts))
    times = np.arange(maxlen)
    xts = np.pad(xts,(0,maxlen-len(xts)),"constant",constant_values=np.nan)
    yts = np.pad(yts,(0,maxlen-len(yts)),"constant",constant_values=np.nan)

    ax.plot(times, xts, color='k')
    ax.plot(times, yts + offset)

    idx = np.linspace(0, len(index1) - 1).astype(int)

    col = []
    for i in idx:
        col.append([(index1[i], xts[index1[i]]),
                    (index2[i], offset + yts[index2[i]])])

    lc = mc.LineCollection(col, linewidths=1, linestyles=":", colors="gray")
    ax.add_collection(lc)
    return ax


lm = LetterMatcher()
lm.run()
