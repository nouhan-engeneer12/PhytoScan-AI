import os
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="PhytoScan AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- Language ----------
if "lang" not in st.session_state:
    st.session_state.lang = "EN"

lang = st.session_state.lang
ar = lang == "AR"

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #F5F4EC;
    color: #193B2B;
}
html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif;
}
.block-container {
    max-width: 1120px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
.hero {
    background: linear-gradient(135deg, #153D2B, #286344);
    color: #FFFFFF;
    padding: 42px 34px;
    border-radius: 26px;
    margin-bottom: 24px;
}
.hero h1 {
    color: #FFFFFF;
    font-size: 2.6rem;
    font-weight: 800;
    margin-bottom: 8px;
}
.hero p {
    color: #E5F0E7;
    font-size: 1.05rem;
}
.panel {
    background: #FFFFFF;
    border: 1px solid #E0E8DE;
    border-radius: 20px;
    padding: 24px;
    margin: 12px 0;
}
.result {
    background: #E8F2E8;
    border-left: 5px solid #397A4D;
    padding: 22px;
    border-radius: 14px;
    margin-top: 18px;
}
.tip {
    background: #F0EBDD;
    padding: 18px;
    border-radius: 14px;
    margin: 10px 0;
}
.small-note {
    color: #627266;
    font-size: 0.9rem;
}
div.stButton > button {
    background: #24583D;
    color: white;
    border: 0;
    border-radius: 12px;
    min-height: 45px;
    font-weight: 700;
    width: 100%;
}
div.stButton > button:hover {
    background: #34744F;
    color: white;
    border: 0;
}
h2, h3 {
    color: #193B2B;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- Translations ----------
T = {
    "EN": {
        "tag": "SMART AGRICULTURE • POWERED BY AI",
        "title": "PhytoScan AI",
        "subtitle": "A smarter first look at tomato leaf health.",
        "intro": "Upload a clear tomato leaf photo to explore the model's prediction and practical care guidance.",
        "upload": "Upload a tomato leaf image",
        "analyze": "Analyze Leaf",
        "another": "Analyze Another Image",
        "result": "Analysis Result",
        "prediction": "Predicted class",
        "confidence": "Model score",
        "symptoms": "Typical symptoms",
        "advice": "Suggested next steps",
        "healthy": "Healthy Tomato Leaf",
        "bacterial": "Bacterial Spot",
        "early": "Early Blight",
        "late": "Late Blight",
        "guide": "How to get a better result",
        "tip1": "Use a sharp, well-lit photo.",
        "tip2": "Keep the leaf centered and visible.",
        "tip3": "Avoid blurry photos and busy backgrounds.",
        "disease_guide": "Plant Care Guide",
        "guide_text": "Check both sides of the leaf, monitor changes, and seek advice from an agricultural specialist if symptoms spread.",
        "about": "About the project",
        "about_text": "PhytoScan AI is a student project exploring deep learning for preliminary tomato leaf disease classification.",
        "warning": "Important limitation",
        "warning_text": "This model recognizes only four tomato-leaf classes. Its predictions may be wrong. The displayed score is not a guarantee of accuracy or a confirmed diagnosis.",
        "missing": "Model file not found. Ensure PhytoScan_AI.keras is in the same repository as this app.",
        "healthy_sym": "The model classified this image as a healthy tomato leaf.",
        "healthy_advice": "Continue monitoring the plant and maintain good growing conditions.",
        "bacterial_sym": "Small dark spots may develop on leaves, sometimes with yellow edges.",
        "bacterial_advice": "Avoid wetting foliage when watering. Remove severely affected material where appropriate and ask an agricultural specialist for guidance.",
        "early_sym": "Brown spots may develop concentric rings and surrounding yellowing.",
        "early_advice": "Monitor nearby leaves, avoid overhead watering, and seek agricultural advice if symptoms continue.",
        "late_sym": "Dark, spreading lesions may occur and can worsen in favorable conditions.",
        "late_advice": "Inspect the plant promptly and seek professional agricultural guidance before choosing any treatment.",
        "preview": "Image preview",
        "working": "Analyzing image...",
        "error": "The image could not be processed. Try another JPG or PNG image.",
        "footer": "Built as an educational AI prototype • Tomato leaves only"
    },
    "AR": {
        "tag": "الزراعة الذكية • مدعوم بالذكاء الاصطناعي",
        "title": "PhytoScan AI",
        "subtitle": "نظرة أولية أذكى على صحة أوراق الطماطم.",
        "intro": "ارفعي صورة واضحة لورقة طماطم لاستكشاف توقع النموذج والحصول على إرشادات عامة للعناية بالنبات.",
        "upload": "ارفعي صورة ورقة طماطم",
        "analyze": "حلّل الورقة",
        "another": "حلّل صورة أخرى",
        "result": "نتيجة التحليل",
        "prediction": "الفئة المتوقعة",
        "confidence": "درجة النموذج",
        "symptoms": "الأعراض المعتادة",
        "advice": "الخطوات المقترحة",
        "healthy": "ورقة طماطم سليمة",
        "bacterial": "التبقع البكتيري",
        "early": "اللفحة المبكرة",
        "late": "اللفحة المتأخرة",
        "guide": "كيف تحصلين على نتيجة أفضل؟",
        "tip1": "استخدمي صورة واضحة وإضاءة جيدة.",
        "tip2": "اجعلي الورقة في منتصف الصورة وظاهرة بوضوح.",
        "tip3": "تجنبي الصور المهزوزة والخلفيات المزدحمة.",
        "disease_guide": "دليل العناية بالنبات",
        "guide_text": "افحصي جانبي الورقة وراقبي التغيرات، واستشيري مهندسًا زراعيًا إذا استمرت الأعراض أو انتشرت.",
        "about": "عن المشروع",
        "about_text": "PhytoScan AI مشروع طلابي يستكشف استخدام التعلم العميق للتصنيف المبدئي لأمراض أوراق الطماطم.",
        "warning": "تنبيه مهم",
        "warning_text": "النموذج يتعرف على أربع فئات فقط من أوراق الطماطم، وقد يخطئ. درجة النموذج ليست ضمانًا لصحة النتيجة ولا تشخيصًا مؤكدًا.",
        "missing": "ملف النموذج غير موجود. تأكدي من وجود PhytoScan_AI.keras في نفس مستودع التطبيق.",
        "healthy_sym": "صنّف النموذج الصورة على أنها ورقة طماطم سليمة.",
        "healthy_advice": "استمري في متابعة النبات والحفاظ على ظروف زراعة مناسبة.",
        "bacterial_sym": "قد تظهر بقع داكنة صغيرة على الأوراق، وأحيانًا تحيط بها حواف صفراء.",
        "bacterial_advice": "تجنبي تبليل الأوراق أثناء الري. أزيلي الأجزاء شديدة الإصابة عند ملاءمة ذلك، واستشيري مهندسًا زراعيًا.",
        "early_sym": "قد تظهر بقع بنية ذات حلقات متداخلة مع اصفرار حولها.",
        "early_advice": "راقبي الأوراق المجاورة، وتجنبي الري فوق الأوراق، واطلبي المشورة الزراعية إذا استمرت الأعراض.",
        "late_sym": "قد تظهر بقع داكنة منتشرة وقد تتفاقم في الظروف المناسبة للمرض.",
        "late_advice": "افحصي النبات سريعًا واستشيري مختصًا زراعيًا قبل اختيار أي علاج.",
        "preview": "معاينة الصورة",
        "working": "جارٍ تحليل الصورة...",
        "error": "تعذرت معالجة الصورة. جرّبي صورة أخرى بصيغة JPG أو PNG.",
        "footer": "نموذج تعليمي لمشروع ذكاء اصطناعي • أوراق الطماطم فقط"
    }
}

t = T[lang]

# ---------- Header ----------
left, right = st.columns([5, 1])
with left:
    st.markdown(
        f'<div class="small-note">{t["tag"]}</div>',
        unsafe_allow_html=True
    )
with right:
    if st.button("العربية" if lang == "EN" else "English"):
        st.session_state.lang = "AR" if lang == "EN" else "EN"
        st.rerun()

st.markdown(f"""
<div class="hero" style="text-align:{'right' if ar else 'left'}">
    <h1>🌿 {t["title"]}</h1>
    <p><b>{t["subtitle"]}</b></p>
    <p>{t["intro"]}</p>
</div>
""", unsafe_allow_html=True)

# ---------- Model ----------
MODEL_PATH = "PhytoScan_AI.keras"

CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy"
]

DISPLAY_NAMES = {
    "Tomato___Bacterial_spot": t["bacterial"],
    "Tomato___Early_blight": t["early"],
    "Tomato___Late_blight": t["late"],
    "Tomato___healthy": t["healthy"]
}

DETAILS = {
    "Tomato___Bacterial_spot": (t["bacterial_sym"], t["bacterial_advice"]),
    "Tomato___Early_blight": (t["early_sym"], t["early_advice"]),
    "Tomato___Late_blight": (t["late_sym"], t["late_advice"]),
    "Tomato___healthy": (t["healthy_sym"], t["healthy_advice"])
}

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error(t["missing"])
    st.stop()

# ---------- Upload and prediction ----------
st.markdown(f"### 📷 {t['upload']}")

uploaded_file = st.file_uploader(
    t["upload"],
    type=["jpg", "jpeg", "png"],
    key="leaf_upload"
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown(f"#### {t['preview']}")
        st.image(image, use_container_width=True)

        if st.button(t["analyze"], key="analyze_leaf"):
            with st.spinner(t["working"]):
                resized = image.resize((128, 128))
                array = np.array(resized, dtype=np.float32) / 255.0
                array = np.expand_dims(array, axis=0)

                model = get_model()
                predictions = np.asarray(model.predict(array, verbose=0))[0]
                index = int(np.argmax(predictions))
                class_name = CLASS_NAMES[index]
                score = float(predictions[index]) * 100
                symptoms, advice = DETAILS[class_name]

            st.markdown(f"## 🌱 {t['result']}")
            st.markdown(f"""
            <div class="result">
                <h3>{DISPLAY_NAMES[class_name]}</h3>
                <p><b>{t["confidence"]}:</b> {score:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(min(max(score / 100, 0.0), 1.0))

            st.markdown(f"""
            <div class="panel">
                <h3>🔎 {t["symptoms"]}</h3>
                <p>{symptoms}</p>
                <hr>
                <h3>🌿 {t["advice"]}</h3>
                <p>{advice}</p>
            </div>
            """, unsafe_allow_html=True)

            st.info(t["warning_text"])

            st.session_state["has_result"] = True

    except Exception:
        st.error(t["error"])

if st.session_state.get("has_result", False):
    if st.button(t["another"], key="another_image"):
        st.session_state.pop("has_result", None)
        st.rerun()

# ---------- Guidance ----------
st.markdown("---")
st.markdown(f"## 📸 {t['guide']}")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="tip">☀️<br><b>{t["tip1"]}</b></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="tip">🍃<br><b>{t["tip2"]}</b></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="tip">📷<br><b>{t["tip3"]}</b></div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="panel">
    <h3>🌱 {t["disease_guide"]}</h3>
    <p>{t["guide_text"]}</p>
</div>
<div class="panel">
    <h3>💡 {t["about"]}</h3>
    <p>{t["about_text"]}</p>
</div>
""", unsafe_allow_html=True)

st.warning(t["warning_text"])

st.markdown("---")
st.markdown(
    f'<p class="small-note" style="text-align:center">{t["footer"]}</p>',
    unsafe_allow_html=True
)
