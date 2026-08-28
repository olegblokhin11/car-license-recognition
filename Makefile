.PHONY: install test smoke prepare autolabel train predict ocr infer

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -U pip
	. .venv/bin/activate && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
	. .venv/bin/activate && pip install -e '.[ocr,detection,autolabel,dev]'

test:
	.venv/bin/python -m pytest tests/ -q

smoke:
	.venv/bin/python scripts/smoke_test.py --images data/raw/images/test --n 20

prepare-data:
	.venv/bin/python scripts/prepare_data.py --data-dir data/raw/images \
		--annotation data/raw/annotation/train_annot.txt --out datasets/dataset_for_yolo

predict:
	.venv/bin/python scripts/predict_detector.py --weights assets/base_ext_v2/best.pt \
		--images data/raw/images/test --out out/predictions.txt --conf 0.25 --imgsz 960

ocr:
	.venv/bin/python scripts/run_ocr.py --predictions out/predictions.txt \
		--images data/raw/images --out out/recognition_results.txt