
import os
import tqdm
import numpy
import librosa
import pyworld

def preprocess(power=4, low=27.5):

    skips = []

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
        tables = []
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
                if data[0][2:-1].isnumeric() or data[0] == "[#TRACKEND]":
                    tables.append([length, length / tempo])
        tables = numpy.array(tables)
        tables = numpy.cumsum(tables, 0)

        # Resample data to ticks
        ticks = numpy.arange(int(positions[-1]))
        pitches = numpy.interp(ticks, numpy.interp(times, tables[:, 1], tables[:, 0]), pitches)
        notes = numpy.interp(ticks, positions, notes)
        pauses = numpy.where(notes < 0, 1, 0)

        # Check audio ust alignment
        template_a = numpy.log2(numpy.maximum(pitches, 55) / 440) * 12 + 69
        template_b = numpy.maximum(notes, 33)
        corr = numpy.corrcoef(template_a, template_b)[0, 1]
        if corr < .5:
            skips.append(audio)
            continue

        # Resample data to ticks
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

        # Create mask
        mask = numpy.ones_like(pitches)
        for xa, xb in silences:
            for part in parts:
                if xa < part[0] or part[1] < xa:
                    continue
                if xb < part[0] or part[1] < xb:
                    continue
                mask[xa - 10:xb + 10] = 0
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
        mask = numpy.pad(mask, (0, pad), constant_values=1)
        bounds[-pad] = 1
        data = (pitches, notes, deltas, bounds, pauses, stable, embeds, normal, mask)
        data = numpy.stack(data)
        
        # Export data
        os.makedirs("npys", exist_ok=True)
        counter = 1
        for i in range(0, len(pitches) - 2048, 1024):
            if numpy.sum(1 - pauses[i:i + 2048]) < 0.5:
                continue
            numpy.save(f"npys/{name}_{counter}.npy", data[:, i:i + 2048])
            counter += 1

    if len(skips) > 0:
        print("Low audio-ust alignment score detected on these files (skipped):")
        for skip in skips:
            print(skip)

if __name__ == "__main__":
    preprocess()