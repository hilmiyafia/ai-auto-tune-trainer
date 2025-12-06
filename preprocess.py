
import os
import tqdm
import numpy
import librosa
import pyworld

def preprocess(power=4, low=27.5):

    for audio in tqdm.tqdm(os.listdir("audios")):
        
        # Read WAV
        data, rate = librosa.load(f"audios/{audio}")
        pitches, times = pyworld.harvest(data.astype(numpy.double), rate)
        name = audio.split(".")[0]

        # Read UST
        tempo = None
        parts = [[0, 0]]
        positions = [0]
        embeds = [0]
        normal = [0]
        notes = [60]
        bounds = [0]
        with open(f"usts/{name}.ust") as file:
            lyric = ""
            length = 0
            for line in file:
                data = line.strip("\n").split("=")
                if data[0] == "Tempo":
                    tempo = 480 * float(data[1]) / 60 / 5
                if data[0] == "Length":
                    length = float(data[1]) / 5
                if data[0] == "NoteNum":
                    positions += [positions[-1], positions[-1] + length]
                    if lyric == "r":
                        notes += [-1, -1]
                    else:
                        notes += [float(data[1]), float(data[1])]
                    bounds += [bounds[-1] + 1, bounds[-1] + 1]
                    embeds += [0, length]
                    normal += [0, 1]
                if data[0] == "Lyric":
                    lyric = data[1].lower()
                    if lyric == "r":
                        if parts[-1][1] > parts[-1][0]:
                            parts.append([parts[-1][1], parts[-1][1]])
                        parts[-1][0] = parts[-1][1] + length
                    parts[-1][1] += length

        # Resample data to ticks
        ticks = numpy.arange(int(positions[-1]))
        pitches = numpy.interp(ticks, times * tempo, pitches)
        notes = numpy.interp(ticks, positions, notes)
        pauses = numpy.where(notes < 0, 1, 0)
        deltas = (notes[1:] - notes[:-1]) * (notes[1:] > 0) * (notes[:-1] > 0)
        deltas = numpy.pad(deltas, (1, 0))
        bounds = numpy.interp(ticks, positions, bounds)
        bounds = numpy.pad(bounds[1:] - bounds[:-1], (1, 0))
        stable = numpy.convolve(bounds, numpy.ones(32), "same")
        stable = numpy.where(stable > 0.5, 0, 1)
        embeds = numpy.interp(ticks, positions, embeds) * (1 - pauses)
        normal = numpy.interp(ticks, positions, normal) * (1 - pauses)
        
        # Detect silences
        silences = [[0, 0]]
        for i in range(len(pitches)):
            if pitches[i] > low:
                if silences[-1][1] > silences[-1][0]:
                    silences.append([silences[-1][1], silences[-1][1]])
                silences[-1][0] = silences[-1][1] + 1
            silences[-1][1] += 1

        # Interpolate silences in parts
        powers = numpy.arange(power)[None]
        for xa, xb in silences:
            for part in parts:
                if xa < part[0] or part[1] < xa:
                    continue
                if xb < part[0] or part[1] < xb:
                    continue
                x1, x2, x3, x4 = xa - 10, xa - 5, xb + 5, xb + 10
                xs = ticks[x1:x4]
                ys = pitches[x1:x4]
                dxs = numpy.concatenate((ticks[x1:x2], ticks[x3:x4])) - xs[0]
                dys = numpy.concatenate((pitches[x1:x2], pitches[x3:x4]))
                txs = (xs - xs[0])[:, None] ** powers
                tys = txs @ numpy.linalg.pinv(dxs[:, None] ** powers) @ dys
                pitches[x1:x4] = tys
                break

        # Pad data
        pad = int((int(len(pitches) / 2048) + 1.5) * 2048) - len(pitches)
        pitches = numpy.maximum(numpy.pad(pitches, (0, pad)), low)
        pitches = numpy.log2(pitches / 220) * 12 + 57
        notes = numpy.pad(notes, (0, pad))
        notes = numpy.maximum(notes, numpy.log2(low / 220) * 12 + 57)
        deltas = numpy.pad(deltas, (0, pad))
        bounds = numpy.pad(bounds, (0, pad))
        pauses = numpy.pad(pauses, (0, pad))
        stable = numpy.pad(stable, (0, pad))
        embeds = numpy.pad(embeds, (0, pad))
        normal = numpy.pad(normal, (0, pad)) * 100
        bounds[-pad] = 1
        data = (pitches, notes, deltas, bounds, pauses, stable, embeds, normal)
        data = numpy.stack(data)
        
        # Export data
        os.makedirs("npys", exist_ok=True)
        counter = 1
        for i in range(0, len(pitches) - 2048, 1024):
            if numpy.sum(1 - pauses[i:i + 2048]) < 0.5:
                continue
            numpy.save(f"npys/{name}_{counter}.npy", data[:, i:i + 2048])
            counter += 1
