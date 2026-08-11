UV ?= uv

install:
	$(UV) sync

# Full pipeline: read metadata, convert MP3, separate the piano stem, apply the
# noise gate, transcribe and quantize MIDI, normalize MIDI, then write results.
main:
	$(UV) run python -m muvisual_workflow.pipeline

separate:
	$(UV) run muvisual-separate

gate:
	$(UV) run muvisual-gate

transcribe:
	$(UV) run muvisual-transcribe

fix-midi:
	$(UV) run muvisual-fix-midi

split-midi:
	$(UV) run muvisual-split-midi

full: separate gate transcribe fix-midi

clean:
	rmdir /s /q data\develop\stem 2>nul || true
	rmdir /s /q data\develop\stem_gated 2>nul || true
	rmdir /s /q data\develop\midi_fixed 2>nul || true
	rmdir /s /q data\midi_hand_split 2>nul || true
	rmdir /s /q data\develop\midi 2>nul || true
