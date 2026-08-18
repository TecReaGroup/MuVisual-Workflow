UV ?= uv
VPS_SYNC_DIR ?= vps-sync

.PHONY: push pull

install:
	$(UV) sync

push:
	$(MAKE) -C $(VPS_SYNC_DIR) upload

pull:
	$(MAKE) -C $(VPS_SYNC_DIR) download

hf:
	$(UV) hf auth login

modal-setup:
	$(UV) run modal setup

# 生产环境部署
modal-deploy:
	$(UV) run modal deploy deploy/modal/app.py
	$(UV) run modal run deploy/modal/app.py --warmup-only

# 生产环境运行测试
modal-api:
	$(UV) run python -m deploy.modal.api

# 开发环境测试部署
modal:
	$(UV) run modal run deploy/modal/app.py

# Full pipeline: read audio tags, copy the original, detect beats once, and
# separate all stems. Store every stem as MP3, gate/transcribe only configured
# instruments, then normalize and quantize every generated MIDI file.
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

quantize-midi:
	$(UV) run muvisual-quantize-midi

split-midi:
	$(UV) run muvisual-split-midi

full:
	$(MAKE) separate
	$(MAKE) gate
	$(MAKE) transcribe
	$(MAKE) fix-midi
	$(MAKE) quantize-midi

clean:
	rmdir /s /q data\develop\stem 2>nul || true
	rmdir /s /q data\develop\stem_gated 2>nul || true
	rmdir /s /q data\develop\midi_fixed 2>nul || true
	rmdir /s /q data\develop\midi_quantized 2>nul || true
	rmdir /s /q data\midi_hand_split 2>nul || true
	rmdir /s /q data\develop\midi 2>nul || true
