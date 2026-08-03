install:
	$(conda) create -n muvisual python=3.12 -y
	$(conda) activate muvisual
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install torch torchvision torchaudio --index-url <https://download.pytorch.org/whl/cu130>
	$(PYTHON) -m pip install -r requirements.txt

separate:
	$(PYTHON) separate_audio.py

gate:
	$(PYTHON) gate_stems.py

transcribe:
	$(PYTHON) transcribe_piano.py

fix-midi:
	$(PYTHON) fix_midi.py

split-midi:
	$(PYTHON) split_midi_voices.py

full: separate gate transcribe fix-midi

clean:
	rmdir /s /q data\stem 2>nul || true
	rmdir /s /q data\stem_gated 2>nul || true
	rmdir /s /q data\midi_fixed 2>nul || true
	rmdir /s /q data\midi_hand_split 2>nul || true
	rmdir /s /q data\midi 2>nul || true
