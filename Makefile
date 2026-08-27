UV ?= uv
VPS_SYNC_DIR ?= vps-sync

.PHONY: install hf run push pull modal-setup modal-deploy modal-api modal

install:
	$(UV) sync

hf:
	$(UV) hf auth login

# Full pipeline
run:
	$(UV) run python -m muvisual_workflow.workflow.pipeline

# Upload files to VPS
push:
	$(MAKE) -C $(VPS_SYNC_DIR) upload

# Download files from VPS
pull:
	$(MAKE) -C $(VPS_SYNC_DIR) download

# Model deployment
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