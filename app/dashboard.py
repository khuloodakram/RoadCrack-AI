"""Arabic Streamlit client for the RoadCrack API."""
import os
from collections import Counter
from pathlib import Path

import httpx
import streamlit as st

API = os.getenv("ROADCRACK_API", "http://127.0.0.1:8000")
st.set_page_config(page_title="RoadCrack AI", page_icon="◈", layout="wide")
st.markdown("""
<style>
[data-testid=stAppViewContainer] { background: #f5f5ee; color: #202c24; }
[data-testid=stSidebar] { background: #e5e9de; }
[data-testid=stMainBlockContainer] { max-width: 1180px; padding-top: 2.6rem; }
.road-hero { background: #19362d; color: #f9fbf1; border-radius: 18px;
  padding: 36px 44px; margin: 0 0 28px; border-bottom: 5px solid #c5df77; }
.road-hero h1 { font-size: clamp(2.5rem, 7vw, 4.8rem); margin: 0; color: #f9fbf1; }
.road-hero p { margin: 4px 0 0; font-size: 1.15rem; color: #d8e7d7; }
</style>
<div class="road-hero"><h1>RoadCrack AI</h1>
<p dir="rtl">منصة تحليل تشققات الطرق</p></div>
""", unsafe_allow_html=True)
st.session_state.setdefault("history", [])


def call(method, route, **kwargs):
    try:
        response = httpx.request(method, API + route, timeout=120, **kwargs)
        if response.is_success:
            return response
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = None
        st.error(detail if isinstance(detail, str) else "تعذر تنفيذ الطلب")
    except httpx.HTTPError:
        st.error("تعذر الاتصال بخدمة التحليل")
    return None


def download(job, file, label, display=None):
    response = call("GET", f"/artifacts/{job}/{file}")
    if response is not None:
        if display:
            display(response.content)
        st.download_button(label, response.content, file_name=file, key=f"{job}-{file}")


with st.sidebar:
    st.header("مساحة العمل")
    page = st.radio("القسم", ["الرئيسية", "تحليل الصور", "تحليل الفيديو", "السجل", "تقييم النماذج"],
                    label_visibility="collapsed")

if page == "الرئيسية":
    st.header("ابدأ تحليل الطريق")
    st.markdown("اختر **تحليل الصور** لاستخراج مواضع العيوب، أو **تحليل الفيديو** لفحص إطارات مختارة وتصدير فيديو النتائج.")
    st.markdown("يتطلب الوضع المفصّل نموذج تقسيم مدرباً. تظهر النتائج بإحداثيات البكسل.")
elif page == "تقييم النماذج":
    st.header("تقييم النماذج")
    report = Path("reports/evaluation.json")
    if report.is_file():
        st.json(report.read_text())
    else:
        st.info("لم تُسجل نتائج تقييم على بيانات مستقلة بعد.")
elif page == "السجل":
    st.header("تحاليل هذه الجلسة")
    if st.session_state.history:
        st.dataframe(st.session_state.history, hide_index=True, use_container_width=True)
    else:
        st.info("لا توجد تحاليل حتى الآن.")
else:
    is_image = page == "تحليل الصور"
    st.header(page)
    health = call("GET", "/health")
    models = health.json()["models"] if health is not None else []
    settings = st.columns(3)
    model = settings[0].selectbox("النموذج", models)
    mode = settings[1].selectbox("وضع التحليل", ["fast", "detailed"],
                                format_func=lambda key: {"fast": "سريع", "detailed": "مفصّل"}[key])
    confidence = settings[2].slider("الحد الأدنى للثقة", 0., 1., .25)
    sample = 1 if is_image else st.number_input("تحليل إطار واحد كل", min_value=1, max_value=120, value=5)
    upload = st.file_uploader("ملف الطريق", type=["jpg", "jpeg", "png", "webp"] if is_image else
                              ["mp4", "mov", "avi", "webm", "mkv"])
    if st.button("بدء التحليل", type="primary", disabled=not (upload and model)):
        data = dict(model=model, mode=mode, confidence=confidence)
        if not is_image:
            data["sample"] = sample
        route = "/images" if is_image else "/videos"
        with st.spinner("جارٍ إرسال الملف…"):
            response = call("POST", route, files={"file": (upload.name, upload, upload.type)}, data=data)
        if response is not None:
            result = response.json()
            st.session_state["image_result" if is_image else "video_id"] = result if is_image else result["id"]
            st.session_state.history.append({"id": result["id"], "الملف": upload.name, "النموذج": model, "النوع": page})
    if is_image and (result := st.session_state.get("image_result")):
        job = result["id"]
        metrics = st.columns(3)
        metrics[0].metric("الاكتشافات", len(result["detections"]))
        metrics[1].metric("الأبعاد", f'{result["width"]} × {result["height"]}')
        metrics[2].metric("زمن المعالجة", f'{result["processing_seconds"]:.2f} s')
        original, overlay = st.columns(2)
        with original:
            st.subheader("الصورة الأصلية")
            download(job, "original.png", "تنزيل الأصل", st.image)
        with overlay:
            st.subheader("النتائج")
            download(job, "overlay.png", "تنزيل الصورة المعالجة", st.image)
        counts = Counter(row["class"] for row in result["detections"])
        if counts:
            st.bar_chart(dict(sorted(counts.items())))
        if result["segmentation"] is not None:
            st.subheader("قناع التقسيم")
            download(job, "mask.png", "تنزيل القناع", st.image)
        for file, label in (("analysis.json", "JSON"), ("detections.csv", "CSV"), ("report.pdf", "PDF")):
            download(job, file, f"تصدير {label}")
    if not is_image and (job := st.session_state.get("video_id")):
        st.button("تحديث الحالة")
        if st.button("إيقاف المعالجة"):
            call("DELETE", f"/videos/{job}")
        response = call("GET", f"/videos/{job}")
        if response is not None:
            result = response.json()
            st.write({"queued": "في الانتظار", "running": "جارٍ التحليل", "done": "اكتمل",
                      "failed": "تعذرت المعالجة", "cancelled": "أُلغي التحليل"}.get(result["status"]))
            if result["status"] == "failed":
                st.error(result["detail"])
            if result["status"] == "done":
                cols = st.columns(3)
                cols[0].metric("الإطارات المحللة", result["analyzed_frames"])
                cols[1].metric("إطارات المعالجة في الثانية", f'{result["processing_fps"]:.2f}')
                cols[2].metric("زمن المعالجة", f'{result["processing_seconds"]:.2f} s')
                download(job, "annotated.mp4", "تنزيل الفيديو", st.video)
                download(job, "frames.jsonl", "تنزيل نتائج الإطارات")
                for file, label in (("analysis.json", "JSON"), ("detections.csv", "CSV"), ("report.pdf", "PDF")):
                    download(job, file, f"تصدير {label}")
