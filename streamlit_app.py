import os
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="PhytoScan AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Language ----------
if "language" not in st.session_state:
    st.session_state.language = "AR"

ar = st.session_state.language == "AR"

T = {
    "AR": {
        "home": "الرئيسية",
        "scanner": "تحليل الصور",
        "library": "دليل الأمراض",
        "care": "دليل العناية",
        "about": "عن المشروع",
        "language": "English",
        "tag": "الزراعة الذكية • الذكاء الاصطناعي",
        "welcome": "مرحبًا بك في",
        "subtitle": "اكتشف صحة النباتات بصورة أذكى",
        "description": "منصة تعليمية تستخدم الذكاء الاصطناعي لتحليل صور الأوراق وتقديم توقع مبدئي وإرشادات عامة للعناية بالنبات.",
        "start": "ابدأ تحليل صورة",
        "explore": "استكشف دليل الأمراض",
        "scanner_title": "محلل الصور الذكي",
        "scanner_desc": "ارفع صورة واضحة لورقة نبات لبدء التحليل.",
        "upload": "اختر صورة من جهازك",
        "analyze": "ابدأ التحليل",
        "again": "حلّل صورة أخرى",
        "result": "نتيجة التحليل",
        "prediction": "الفئة المتوقعة",
        "score": "درجة النموذج",
        "symptoms": "الأعراض الشائعة",
        "advice": "إرشادات عامة",
        "healthy": "ورقة طماطم سليمة",
        "bacterial": "التبقع البكتيري",
        "early": "اللفحة المبكرة",
        "late": "اللفحة المتأخرة",
        "tip_title": "كيف تحصل على صورة أفضل؟",
        "tip1": "استخدم إضاءة جيدة.",
        "tip2": "اجعل الورقة واضحة وفي منتصف الصورة.",
        "tip3": "تجنب الصور المهزوزة والخلفيات المزدحمة.",
        "library_title": "دليل أمراض أوراق الطماطم",
        "library_desc": "هذه هي الفئات الأربع التي يستطيع النموذج الحالي توقعها.",
        "care_title": "دليل العناية بالنبات",
        "care1": "الري: تجنب الإفراط في الري، وراعِ احتياجات النبات وظروف التربة.",
        "care2": "الإضاءة: وفر ضوءًا مناسبًا لنوع النبات دون تعريضه لظروف قاسية.",
        "care3": "الفحص: افحص جانبي الأوراق بانتظام وراقب أي تغيرات جديدة.",
        "care4": "النظافة: نظف أدوات الزراعة، وتجنب نقل الأجزاء المصابة بين النباتات.",
        "care5": "الاستشارة: عند انتشار الأعراض، استشر مهندسًا زراعيًا قبل اختيار العلاج.",
        "about_title": "عن PhytoScan AI",
        "about_desc": "مشروع طلابي يستكشف توظيف التعلم العميق في التصنيف المبدئي لأمراض أوراق الطماطم.",
        "step1": "1. رفع الصورة",
        "step2": "2. تجهيز الصورة",
        "step3": "3. تحليل النموذج",
        "step4": "4. عرض التوقع",
        "limitation": "حدود النموذج",
        "limitation_text": "النموذج الحالي مدرب على أربع فئات لأوراق الطماطم فقط. قد يخطئ في التوقع، ولا يمثل بديلًا عن التشخيص الزراعي المتخصص.",
        "missing": "ملف النموذج غير موجود. تأكد من وجود PhytoScan_AI.keras في مستودع المشروع.",
        "error": "تعذر تحليل الصورة. جرّب صورة أخرى بصيغة JPG أو PNG.",
        "wait": "جارٍ تحليل الصورة...",
        "footer": "PhytoScan AI • نموذج تعليمي للكشف المبدئي عن أمراض أوراق الطماطم",
        "bacterial_sym": "قد تظهر بقع داكنة صغيرة على الأوراق، وقد تحيط بها حواف صفراء.",
        "bacterial_advice": "راقب انتشار البقع، وتجنب تبليل الأوراق أثناء الري، واستشر مختصًا زراعيًا عند الحاجة.",
        "early_sym": "قد تظهر بقع بنية ذات حلقات متداخلة، مع اصفرار حول المنطقة المصابة.",
        "early_advice": "راقب الأوراق المجاورة وتجنب الري فوق الأوراق، واطلب المشورة الزراعية إذا استمرت الأعراض.",
        "late_sym": "قد تظهر بقع داكنة تنتشر على الأوراق وقد تتفاقم في ظروف مناسبة للمرض.",
        "late_advice": "افحص النبات سريعًا واستشر مختصًا زراعيًا قبل اختيار أي علاج.",
        "healthy_sym": "صنّف النموذج الصورة على أنها ورقة طماطم سليمة.",
        "healthy_advice": "واصل متابعة النبات والحفاظ على ظروف زراعة مناسبة.",
        "general": "لا يمكن للنموذج تأكيد المرض اعتمادًا على الصورة وحدها."
    },
    "EN": {
        "home": "Home",
        "scanner": "AI Scanner",
        "library": "Disease Library",
        "care": "Plant Care",
        "about": "About",
        "language": "العربية",
        "tag": "SMART AGRICULTURE • ARTIFICIAL INTELLIGENCE",
        "welcome": "Welcome to",
        "subtitle": "A smarter look at plant health",
        "description": "An educational platform that analyzes leaf images to provide a preliminary prediction and general plant-care guidance.",
        "start": "Analyze an Image",
        "explore": "Explore Disease Library",
        "scanner_title": "AI Image Scanner",
        "scanner_desc": "Upload a clear leaf image to begin the analysis.",
        "upload": "Choose an image from your device",
        "analyze": "Analyze Leaf",
        "again": "Analyze Another Image",
        "result": "Analysis Result",
        "prediction": "Predicted class",
        "score": "Model score",
        "healthy": "Healthy Tomato Leaf",
        "bacterial": "Bacterial Spot",
        "early": "Early Blight",
        "late": "Late Blight",
        "symptoms": "Common symptoms",
        "advice": "General guidance",
        "tip_title": "How to get a better image",
        "tip1": "Use good lighting.",
        "tip2": "Keep the leaf clear and centered.",
        "tip3": "Avoid blurry photos and busy backgrounds.",
        "library_title": "Tomato Leaf Disease Library",
        "library_desc": "These are the four classes recognized by the current model.",
        "care_title": "Plant Care Guide",
        "care1": "Watering: Avoid overwatering and consider soil and plant needs.",
        "care2": "Light: Provide suitable light for the plant species.",
        "care3": "Inspection: Regularly inspect both sides of leaves.",
        "care4": "Hygiene: Clean gardening tools and avoid transferring affected material.",
        "care5": "Expert advice: Consult an agricultural specialist if symptoms spread.",
        "about_title": "About PhytoScan AI",
        "about_desc": "A student project exploring deep learning for preliminary tomato leaf disease classification.",
        "step1": "1. Upload image",
        "step2": "2. Prepare image",
        "step3": "3. Run model",
        "step4": "4. Show prediction",
        "limitation": "Model limitations",
        "limitation_text": "The current model recognizes only four tomato-leaf classes. Predictions may be wrong and are not a substitute for expert agricultural diagnosis.",
        "missing": "Model file not found. Ensure PhytoScan_AI.keras is in the project repository.",
        "error": "Could not analyze the image. Try another JPG or PNG image.",
        "wait": "Analyzing image...",
        "footer": "PhytoScan AI • Educational prototype for preliminary tomato leaf disease classification",
        "bacterial_sym": "Small dark spots may appear on leaves, sometimes with yellow edges.",
        "bacterial_advice": "Monitor the spots, avoid wetting leaves during watering, and consult an agricultural specialist if needed.",
        "early_sym": "Brown spots may show concentric rings with surrounding yellowing.",
        "early_advice": "Monitor nearby leaves, avoid overhead watering, and seek agricultural advice if symptoms continue.",
        "late_sym": "Dark spreading lesions may appear and can worsen under favorable disease conditions.",
        "late_advice": "Inspect the plant promptly and consult an agricultural specialist before selecting a treatment.",
        "healthy_sym": "The model classified this image as a healthy tomato leaf.",
        "healthy_advice": "Continue monitoring the plant and maintain suitable growing conditions.",
        "general": "The model cannot confirm a disease from an image alone."
    }
}

t = T[st.session_state.language]

# ---------- Design ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #F4F3EB;
    color: #183D2B;
}
html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif;
}
[data-testid="stSidebar"] {
    background: #153B2A;
}
[data-testid="stSidebar"] * {
    color: #F5F5EC !important;
}
.block-container {
    max-width: 1180px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
.hero {
    background: linear-gradient(130deg, #153B2A, #2C6A48);
    border-radius: 26px;
    padding: 42px 34px;
    color: white;
    margin: 16px 0 26px 0;
}
.hero h1 {
    color: white !important;
    font-size: 2.7rem;
    font-weight: 800;
}
.hero p {
    color: #E3EFE5 !important;
    font-size: 1.05rem;
}
.panel {
    background: white;
    border: 1px solid #E0E7DC;
    border-radius: 20px;
    padding: 22px;
    margin: 12px 0;
}
.metric-card {
    background: white;
    border: 1px solid #E0E7DC;
    border-radius: 18px;
    padding: 22px;
    min-height: 125px;
}
.tip {
    background: #E7EFE3;
    border-radius: 16px;
    padding: 18px;
    margin: 8px 0;
}
.result {
    background: #E5F0E5;
    border-left: 5px solid #3B8054;
    padding: 20px;
    border-radius: 14px;
    margin-top: 16px;
}
div.stButton > button {
    border-radius: 12px;
    font-weight: 700;
    min-height: 44px;
    border: 1px solid #24583D;
}
div.stButton > button[kind="primary"] {
    background: #24583D;
    color: white;
}
h1, h2, h3 {
    color: #183D2B;
}
.small-note {
    color: #6B7B6E;
    font-size: .9rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar navigation ----------
with st.sidebar:
    st.markdown("# 🌿 PhytoScan AI")
    st.caption(t["tag"])
    page = st.radio(
        "Navigation" if not ar else "التنقل",
        [t["home"], t["scanner"], t["library"], t["care"], t["about"]],
        label_visibility="collapsed"
    )
    st.markdown("---")
    if st.button(t["language"], use_container_width=True):
        st.session_state.language = "EN" if ar else "AR"
        st.rerun()

# ---------- Home ----------
if page == t["home"]:
    st.markdown(f"""
    <div class="hero" dir="{'rtl' if ar else 'ltr'}">
        <p>{t["tag"]}</p>
        <h1>🌿 {t["welcome"]} PhytoScan AI</h1>
        <h3 style="color:white">{t["subtitle"]}</h3>
        <p>{t["description"]}</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="panel">
            <h3>📷 {t["scanner"]}</h3>
            <p>{t["scanner_desc"]}</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button(t["start"], type="primary", use_container_width=True):
            st.session_state.selected_page = t["scanner"]
            st.rerun()
    with c2:
        st.markdown(f"""
        <div class="panel">
            <h3>📚 {t["library"]}</h3>
            <p>{t["library_desc"]}</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button(t["explore"], use_container_width=True):
            st.session_state.selected_page = t["library"]
            st.rerun()

    st.markdown(f"## 🌱 {t['tip_title']}")
    a, b, c = st.columns(3)
    with a:
        st.markdown(f'<div class="tip">☀️<br><b>{t["tip1"]}</b></div>', unsafe_allow_html=True)
    with b:
        st.markdown(f'<div class="tip">🍃<br><b>{t["tip2"]}</b></div>', unsafe_allow_html=True)
    with c:
        st.markdown(f'<div class="tip">📷<br><b>{t["tip3"]}</b></div>', unsafe_allow_html=True)

# ---------- Model ----------
MODEL_PATH = "PhytoScan_AI.keras"
CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy"
]

if page == t["scanner"]:
    st.title(f"📷 {t['scanner_title']}")
    st.write(t["scanner_desc"])

    if not os.path.exists(MODEL_PATH):
        st.error(t["missing"])
    else:
        uploaded = st.file_uploader(
            t["upload"],
            type=["jpg", "jpeg", "png"],
            key="plant_image"
        )

        if uploaded is not None:
            try:
                image = Image.open(uploaded).convert("RGB")
                st.image(image, caption=t["upload"], use_container_width=True)

                if st.button(t["analyze"], type="primary"):
                    with st.spinner(t["wait"]):
                        resized = image.resize((128, 128))
                        array = np.asarray(resized, dtype=np.float32) / 255.0
                        array = np.expand_dims(array, axis=0)

                        @st.cache_resource
                        def get_model():
                            return load_model(MODEL_PATH)

                        predictions = np.asarray(
                            get_model().predict(array, verbose=0)
                        )[0]
                        idx = int(np.argmax(predictions))
                        cls = CLASS_NAMES[idx]
                        score = float(predictions[idx]) * 100

                        name_key = {
                            CLASS_NAMES[0]: "bacterial",
                            CLASS_NAMES[1]: "early",
                            CLASS_NAMES[2]: "late",
                            CLASS_NAMES[3]: "healthy"
                        }
                        key = name_key[cls]
                        symptom_key = {
                            CLASS_NAMES[0]: "bacterial_sym",
                            CLASS_NAMES[1]: "early_sym",
                            CLASS_NAMES[2]: "late_sym",
                            CLASS_NAMES[3]: "healthy_sym"
                        }
                        advice_key = {
                            CLASS_NAMES[0]: "bacterial_advice",
                            CLASS_NAMES[1]: "early_advice",
                            CLASS_NAMES[2]: "late_advice",
                            CLASS_NAMES[3]: "healthy_advice"
                        }

                    st.markdown(f"## 🌱 {t['result']}")
                    st.markdown(f"""
                    <div class="result">
                        <h3>{t[key]}</h3>
                        <p><b>{t["score"]}:</b> {score:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(min(max(score / 100, 0.0), 1.0))

                    x, y = st.columns(2)
                    with x:
                        st.markdown(f"""
                        <div class="panel">
                            <h3>🔎 {t["symptoms"]}</h3>
                            <p>{t[symptom_key[cls]]}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with y:
                        st.markdown(f"""
                        <div class="panel">
                            <h3>🌿 {t["advice"]}</h3>
                            <p>{t[advice_key[cls]]}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.warning(t["limitation_text"])
                    if st.button(t["again"]):
                        st.rerun()

            except Exception:
                st.error(t["error"])

# ---------- Disease library ----------
if page == t["library"]:
    st.title(f"📚 {t['library_title']}")
    st.write(t["library_desc"])

    diseases = [
        ("bacterial", "bacterial_sym", "bacterial_advice", "🟤"),
        ("early", "early_sym", "early_advice", "🍂"),
        ("late", "late_sym", "late_advice", "🌑"),
        ("healthy", "healthy_sym", "healthy_advice", "🌿")
    ]
    for name, symptom, advice, emoji in diseases:
        with st.expander(f"{emoji} {t[name]}"):
            st.write(f"**{t['symptoms']}:** {t[symptom]}")
            st.write(f"**{t['advice']}:** {t[advice]}")

# ---------- Plant care ----------
if page == t["care"]:
    st.title(f"🌱 {t['care_title']}")
    for key in ["care1", "care2", "care3", "care4", "care5"]:
        st.markdown(f'<div class="panel">{t[key]}</div>', unsafe_allow_html=True)
    st.info(t["general"])

# ---------- About ----------
if page == t["about"]:
    st.title(f"🌿 {t['about_title']}")
    st.markdown(f'<div class="panel"><p>{t["about_desc"]}</p></div>', unsafe_allow_html=True)
    st.subheader(t["about"])
    cols = st.columns(4)
    steps = ["step1", "step2", "step3", "step4"]
    for col, key in zip(cols, steps):
        with col:
            st.markdown(f'<div class="metric-card">{t[key]}</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.subheader(t["limitation"])
    st.warning(t["limitation_text"])

st.markdown("---")
st.markdown(
    f'<p class="small-note" style="text-align:center">{t["footer"]}</p>',
    unsafe_allow_html=True
)
