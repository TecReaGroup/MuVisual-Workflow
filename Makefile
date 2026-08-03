PYTHON ?= python

AUDIO_INPUT ?= data/audio
AUDIO_OUTPUT ?= data/stem
MODEL_DIR ?= data/model/BS-Rofo-SW
MODEL_NAME ?= BS-Roformer-SW.ckpt

STEM_INPUT ?= data/stem
STEM_OUTPUT ?= data/stem_gated

MIDI_INPUT ?= data/midi
MIDI_OUTPUT ?= data/midi_fixed

HAND_SPLIT_INPUT ?= data/midi_fixed
HAND_SPLIT_OUTPUT ?= data/midi_hand_split
PIANO_SVSEP_ROOT ?= $(CURDIR)/piano_svsep
PIANO_SVSEP_MODEL ?= $(PIANO_SVSEP_ROOT)/pretrained_models/model.ckpt

TRANSCRIBE_INPUT ?= data/stem_gated/一生爱你_(piano)_BS-Roformer-SW.wav
TRANSCRIBE_OUTPUT ?= data/midi/一生爱你_(piano)_BS-Roformer-SW.mid

.PHONY: help install install-audio install-transkun separate gate fix-midi split-midi transcribe full clean

help:
	@echo "Available targets:"
	@echo "  make install             Install base Python dependencies"
	@echo "  make install-audio       Install audio separator dependency"
	@echo "  make install-transkun    Install transkun dependency"
	@echo "  make separate            Separate audio into stems"
	@echo "  make gate                Apply noise gate to stems"
	@echo "  make fix-midi            Normalize and fix MIDI files"
	@echo "  make split-midi          Split piano MIDI into hand channels"
	@echo "  make transcribe          Transcribe a piano stem to MIDI"
	@echo "  make full                Run the full workflow"
	@echo "  make clean               Remove generated output directories"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-audio:
	$(PYTHON) -m pip install -U "audio-separator[gpu]"

install-transkun:
	$(PYTHON) -m pip install transkun

separate:
	$(PYTHON) separate_audio.py --input $(AUDIO_INPUT) --output $(AUDIO_OUTPUT) --model-dir $(MODEL_DIR) --model $(MODEL_NAME)

gate:
	$(PYTHON) gate_stems.py --input $(STEM_INPUT) --output $(STEM_OUTPUT)

fix-midi:
	$(PYTHON) fix_midi.py --input $(MIDI_INPUT) --output $(MIDI_OUTPUT)

split-midi:
	$(PYTHON) split_midi_voices.py --input $(HAND_SPLIT_INPUT) --output $(HAND_SPLIT_OUTPUT) --piano-svsep-root $(PIANO_SVSEP_ROOT) --model $(PIANO_SVSEP_MODEL)

transcribe:
	$(PYTHON) transcribe_piano.py --input $(TRANSCRIBE_INPUT) --output $(TRANSCRIBE_OUTPUT)

full: separate gate fix-midi split-midi transcribe

clean:
	rmdir /s /q data\stem 2>nul || true
	rmdir /s /q data\stem_gated 2>nul || true
	rmdir /s /q data\midi_fixed 2>nul || true
	rmdir /s /q data\midi_hand_split 2>nul || true
	rmdir /s /q data\midi 2>nul || true
