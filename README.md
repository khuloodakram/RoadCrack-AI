# RoadCrack-AI

RoadCrack AI detects and classifies visible road damage in images and video and optionally segments crack pixels. It includes a PyTorch training workflow, a FastAPI service and an Arabic Streamlit dashboard. The app needs trained road-damage weights supplied by the operator.

## التشغيل

يتطلب المسار المحلي Python 3.12. يستخدم التثبيت الافتراضي PyTorch للمعالج. شغّل الأوامر من جذر المستودع.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -c requirements.lock.txt
export PYTHONPATH=src
python scripts/check_env.py
python -m pytest
cp .env.example .env
```

عدّل مسارات الأوزان في `.env` قبل التحليل. يشغّل المثال التالي الخدمتين في طرفيتين منفصلتين بعد تنشيط البيئة.

```bash
# Terminal 1
set -a
source .env
set +a
uvicorn app.api:app --port 8000

# Terminal 2
streamlit run app/dashboard.py --server.port 8501 --server.maxUploadSize 95
```

افتح لوحة التحكم على [المنفذ 8501](http://localhost:8501) ووثائق API على [المنفذ 8000](http://localhost:8000/docs).

## English

Use the commands above for a CPU development install. Provide trusted road-damage detector checkpoints in `ROADCRACK_DETECTORS` and optionally a segmentation checkpoint in `ROADCRACK_SEGMENTER`. Fast mode returns boxes and class scores. Detailed mode also returns a binary mask in image coordinates. For a containerized CPU install, run `docker compose up --build` after configuring `.env`.

- [دليل الاستخدام بالعربية](docs/USER_GUIDE_AR.md)
- [Data sources and licenses](docs/DATA_AND_LICENSES.md)
- [Engineering and training guide](docs/ENGINEERING.md)
- [GPU workflow for Colab](notebooks/train_colab.ipynb)
- [Scientific report](docs/SCIENTIFIC_REPORT.md)
- [Progress and verification](docs/PROGRESS.md)
- [Final status](docs/FINAL_STATUS.md)
