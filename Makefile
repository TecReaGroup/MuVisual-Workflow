UV ?= uv

install:
	$(UV) sync

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
	rmdir /s /q data\stem 2>nul || true
	rmdir /s /q data\stem_gated 2>nul || true
	rmdir /s /q data\midi_fixed 2>nul || true
	rmdir /s /q data\midi_hand_split 2>nul || true
	rmdir /s /q data\midi 2>nul || true
