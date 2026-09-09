UV ?= uv
VPS_SYNC_DIR ?= vps-sync

.PHONY: install hf run push pull modal-setup modal-deploy modal-api modal

install:
	$(UV) sync

hf:
	$(UV) run hf auth login --force

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
# 获取 modal 访问权限
modal-setup:
	$(UV) run modal setup

# 生产环境部署
modal-deploy:
	$(UV) run modal run deploy/modal/app.py --warmup-only
	$(UV) run modal deploy deploy/modal/app.py

# 生产环境 API 测试
modal-api:
	$(UV) run python -m deploy.modal.api

# 开发环境原生部署(详细日志便于调试)
modal:
	$(UV) run modal run deploy/modal/app.py
