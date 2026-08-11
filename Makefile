UV ?= uv

install:
	$(UV) sync

# Full pipeline: read audio tags, copy the original, detect beats once, and
# separate all stems. Store every stem as MP3, gate/transcribe only configured
# instruments, then normalize every generated MIDI file.
main:
	$(UV) run python -m muvisual_workflow.workflow.pipeline

separate:
	$(UV) run muvisual-separate

gate:
	$(UV) run muvisual-gate

transcribe:
	$(UV) run muvisual-transcribe

beats:
	$(UV) run muvisual-beats

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
