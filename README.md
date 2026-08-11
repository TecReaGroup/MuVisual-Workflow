# MuVisual-Workflow

## Introduction

MuVisual is a tool for visualizing music.
Now, The default sheet music is in 4/4 time.

## Instrument separation model

The workflow uses the six-stem `BS-Roformer-SW.ckpt` model registered by
`audio-separator`. A local checkpoint is not required: the library downloads
the model and its configuration automatically on the first run and reuses its
cache on subsequent runs.

The first run therefore requires network access. For cloud deployments, the
library's default temporary cache can be used without configuration. To keep
the model between container restarts, create a directory on a persistent
volume and set `AUDIO_SEPARATOR_MODEL_DIR` to that directory.

## Develop

```bash
uv sync
```

## Run the complete pipeline

Place source audio files in `data/input`. Each file must contain `title` and
`album` metadata because the pipeline uses those values for the output folder
and file names.

```bash
make main
```

The `main` target runs these steps for every source file:

1. Read the `title` and `album` metadata and convert the original audio to MP3.
2. Use `BS-Roformer-SW.ckpt` to separate the audio and select the piano stem.
3. Apply the noise gate and convert the processed piano stem to MP3.
4. Transcribe the piano stem to MIDI with Transkun and create a quantized MIDI.
5. Normalize both MIDI files and write the final results to `data/output`.

Each completed output folder contains the original MP3, processed piano MP3,
normalized MIDI, and normalized quantized MIDI. Complete existing outputs are
skipped automatically.

## Run individual stages

Separate the audio files in `data/develop/audio` into six stems:

```bash
uv run muvisual-separate
```

The individual stage commands use working directories under `data/develop`:

```bash
make separate
make gate
make transcribe
make fix-midi
```
