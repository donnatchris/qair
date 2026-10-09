# SOURCE=videos/test-video.mp4
SOURCE=videos/test-qair.mp4
YOLO=.venv/bin/yolo

detect-medium:
	@$(YOLO) predict \
		model=yolo26m.pt \
		source=$(SOURCE) \
		imgsz=1280 \
		conf=0.05 \
		classes=14 \
		device=cpu \
		save=True \
		save_txt=True \
		save_conf=True

detect-tiled:
	@.venv-gpu/bin/python scripts/detect_tiled.py