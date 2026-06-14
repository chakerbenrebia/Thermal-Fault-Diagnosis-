import streamlit as st
import cv2
import torch
import joblib
import numpy as np
from PIL import Image
from torchvision import models, transforms
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
import pandas as pd
# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Thermal Fault Diagnosis",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ Thermal Fault Diagnosis System")
st.write("Upload a thermal image.")

# =====================================================
# LOAD RANDOM FOREST
# =====================================================

MODEL_PATH = "rf_model.pkl"

rf_model = joblib.load(MODEL_PATH)

# =====================================================
# LOAD RESNET18
# =====================================================

weights = models.ResNet18_Weights.DEFAULT

resnet = models.resnet18(weights=weights)

feature_extractor = torch.nn.Sequential(
    *list(resnet.children())[:-1]
)

feature_extractor.eval()

# =====================================================
fault_images = {

    "Rotor-0": "images/rotor_fault.jpg",

    "Fan": "images/fan_fault.jpg",

    "Noload": "images/healthy_motor.jpg",

    "A10": "images/stator_fault.jpg",
    "A30": "images/stator_fault.jpg",
    "A50": "images/stator_fault.jpg",

    "A&C10": "images/stator_fault.jpg",
    "A&C30": "images/stator_fault.jpg",

    "A&C&B10": "images/stator_fault.jpg",
    "A&C&B30": "images/stator_fault.jpg",

    "A&B50": "images/stator_fault.jpg"
}

# TRANSFORM
# =====================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =====================================================
# fault description
# =====================================================

fault_descriptions = {

    "Noload":
    "Healthy motor operating under no-load condition.",

    "Fan":
    "Cooling fan fault causing abnormal thermal distribution.",

    "Rotor-0":
    "Rotor defect causing thermal imbalance and increased losses.",

    "A10":
    "10% stator winding short circuit severity.",

    "A30":
    "30% stator winding short circuit severity.",

    "A50":
    "50% stator winding short circuit severity.",

    "A&C10":
    "10% fault involving phases A and C.",

    "A&C30":
    "30% fault involving phases A and C.",

    "A&B50":
    "Severe fault involving phases A and B.",

    "A&C&B10":
    "Minor multi-phase fault.",

    "A&C&B30":
    "Advanced multi-phase fault."
}
# =====================================================
# MAINTENANCE RECOMMENDATIONS
# =====================================================

recommendations = {

    "Fan":
    "Inspect cooling fan condition. Check airflow, ventilation openings, fan blades and cooling efficiency.",

    "Rotor-0":
    "Inspect rotor bars, shaft alignment, rotor balance and vibration condition.",

    "Noload":
    "Motor operating normally. Continue periodic monitoring and preventive maintenance.",

    "A10":
    "Minor stator winding fault detected. Schedule preventive maintenance and thermal monitoring.",

    "A30":
    "Moderate stator winding degradation detected. Perform insulation testing and corrective maintenance.",

    "A50":
    "Severe stator winding short circuit detected. Immediate intervention recommended.",

    "A&C10":
    "Minor phase-to-phase degradation between phases A and C. Inspect winding insulation.",

    "A&C30":
    "Moderate phase-to-phase fault detected. Maintenance strongly recommended.",

    "A&B50":
    "Severe two-phase fault detected. Immediate corrective action required.",

    "A&C&B10":
    "Minor multi-phase anomaly detected. Increase monitoring frequency.",

    "A&C&B30":
    "Advanced multi-phase fault detected. Immediate diagnostic investigation required."
}

# =====================================================
# PREPROCESS + FEATURE EXTRACTION
# =====================================================

def extract_features(uploaded_image):

    img = np.array(uploaded_image)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY
        )
    else:
        gray = img

    filtered = cv2.medianBlur(
        gray,
        3
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    enhanced = clahe.apply(filtered)

    normalized = enhanced.astype(
        np.float32
    ) / 255.0

    rows, cols = normalized.shape

    row_start = 39
    row_end = min(250, rows)

    col_start = 39
    col_end = min(300, cols)

    roi = normalized[
        row_start:row_end,
        col_start:col_end
    ]

    roi_uint8 = (
        roi * 255
    ).astype(np.uint8)

    threshold = np.max(roi_uint8) * 0.9

    hot_mask = roi_uint8 >= threshold
    hot_mask = hot_mask.astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        hot_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    roi_display = cv2.cvtColor(
        roi_uint8,
        cv2.COLOR_GRAY2RGB
    )

    if len(contours) > 0:

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        (x, y), radius = cv2.minEnclosingCircle(
            largest_contour
        )

        cv2.circle(
            roi_display,
            (int(x), int(y)),
            int(radius),
            (255, 0, 0),
            3
        )

    roi_rgb = cv2.cvtColor(
        roi_uint8,
        cv2.COLOR_GRAY2RGB
    )

    pil_img = Image.fromarray(
        roi_rgb
    )
    img_tensor = transform(
        pil_img
    )

    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():

        features = feature_extractor(
            img_tensor
        )

    features = (
        features
        .squeeze()
        .cpu()
        .numpy()
    )

    return features.reshape(1, -1), roi_display

def get_health_index(fault):

    health_table = {

        "Noload":100,

        "A10":80,

        "A&C10":75,

        "A&C&B10":70,

        "Fan":65,

        "Rotor-0":60,

        "A30":50,

        "A&C30":40,

        "A50":20,

        "A&B50":15,

        "A&C&B30":10
    }

    return health_table.get(
        fault,
        50
    )

# =====================================================
# RISK LEVEL
# =====================================================

def get_risk_level(fault):

    low_faults = [
        "Noload",
        "A10"
    ]

    medium_faults = [
        "Fan",
        "Rotor-0",
        "A&C10",
        "A&C&B10"
    ]

    if fault in low_faults:
        return "🟢 LOW"

    elif fault in medium_faults:
        return "🟡 MEDIUM"

    else:
        return "🔴 HIGH"

def get_fault_category(fault):

    if fault == "Noload":
        return "Healthy Condition"

    elif fault == "Fan":
        return "Cooling Fault"

    elif fault == "Rotor-0":
        return "Rotor Fault"

    else:
        return "Stator Fault"
# =====================================================
# pdf
# =====================================================

def create_pdf_report(
    date,
    fault,
    confidence,
    health,
    risk,
    recommendation,
    original_image_path,
    fault_image_path
):

    buffer = BytesIO()

    pdf = canvas.Canvas(buffer)

    y = 800
    pdf.setTitle(
        "Thermal Fault Diagnosis Report"
    )

    pdf.setFont(
        "Helvetica-Bold",
        18
    )
    pdf.drawString(
        50,
        y,
        "Thermal image"
    )
    y-= 180
    pdf.drawImage(
        original_image_path,
        50,
        y,
        width=200,
        height=160
    )
    pdf.drawString(
        320,
        y + 170,
        "Affected Component"
    )
    pdf.drawImage(
        fault_image_path,
        300,
        y,
        width=200,
        height=160
    )

    pdf.drawString(
        50,
        y,
        "THERMAL FAULT DIAGNOSIS REPORT"
    )

    y -= 10

    pdf.line(
        50,
        y,
        550,
        y
    )

    y -= 40
    pdf.setFont(
      "Helvetica",
        12
    )

    pdf.drawString(
        50,
        y,
        f"Date: {date}"
    )

    y -= 25

    pdf.drawString(
        50,
        y,
        f"Fault: {fault}"
    )

    y -= 25

    pdf.drawString(
        50,
        y,
        f"Confidence: {confidence:.2f}%"
    )

    y -= 25

    pdf.drawString(
        50,
        y,
        f"Health Index: {health}/100"
    )

    y -= 25

    pdf.drawString(
        50,
        y,
        f"Risk Level: {risk}"
    )

    y -= 40

    pdf.drawString(
        50,
        y,
        "Recommendation:"
    )

    y -= 25

    lines = recommendation.split(".")

    for line in lines:

        if line.strip():

            pdf.drawString(
                70,
                y,
                line.strip()
            )

            y -= 20

    pdf.save()

    buffer.seek(0)

    return buffer

# =====================================================
# EQUIPMENT SELECTION
# =====================================================

equipment = st.selectbox(

    "Select Equipment Type",

    [
        "Pump",
        "Induction Motor",
        "Fan ",
        "Turbine",
        "Compressor",
        "Generator",
        "Gearbox ",
        "Transformer"
    ]
)

if equipment == "Induction Motor":

    uploaded_file = st.file_uploader(

        "Upload Thermal Image",

        type=[
            "bmp",
            "jpg",
            "jpeg",
            "png"
        ]
    )

else:

    st.warning(
        "This module is under development. Future versions will support this equipment."
    )

    uploaded_file = None

# =====================================================
# PREDICTION
# =====================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    original_image_path = "temp_thermal.png"
    image.save(
        original_image_path
    )

    features, roi_image = extract_features(image)

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Uploaded Thermal Image",
            width=400
        )

    with col2:

        st.image(
            roi_image,
            caption="Processed ROI",
            width=400
        )

    features = pd.DataFrame(
        features,
        columns=rf_model.feature_names_in_
    )
    prediction = rf_model.predict(
        features
    )[0]
    fault_image_path = fault_images.get(
        prediction,
        "images/healthy_motor.jpg"
    )
    fault_category = get_fault_category(
        prediction
    )
    with col1:
            st.subheader("Inspection Summary")
            current_date = datetime.now().strftime(
                "%d/%m/%Y %H:%M:%S"
            )
            st.markdown(
                f"**Inspection Date:** {current_date}"
            )
            st.markdown(
                f"**Equipment type:** {equipment}"
            )
            st.markdown(
                f"**Fault Category:** {fault_category}"
            )
            image_path = fault_images.get(
                prediction
            )
            if image_path:
                st.image(
                    image_path,
                    caption="affected Component",
                    width=400
                )
    probabilities = rf_model.predict_proba(
        features
    )[0]

    classes = rf_model.classes_

    best_confidence = (
        np.max(probabilities) * 100
    )
    current_date = datetime.now().strftime(
           "%d/%m/%Y %H:%M:%S"
       )

    top_idx = np.argsort(
           probabilities
       )[::-1][:3]

    health = get_health_index(
           prediction
       )

    risk = get_risk_level(
           prediction
       )
    fault_category = get_fault_category(
        prediction
    )

    with col2:

        st.success(
            f"Predicted Fault: {prediction}"
        )
        st.info(
            fault_descriptions.get(
            prediction,
        "Description not available."
              )
        )

        st.metric(
            "Confidence",
            f"{best_confidence:.2f}%"
        )

        st.subheader("Equipment Health")
        st.progress(
            health /100
        )
        st.write(
            f"{health}/100"
        )
        

        st.metric(
            "Risk Level",
            risk
        )

        st.subheader("Top 3 Predictions")

        for idx in top_idx:
            st.progress(float(probabilities[idx]))
            st.write(
                f"{classes[idx]} : "
                f"{probabilities[idx]*100:.2f}%"
            )


    st.header(
        " Maintenance Recommendation"
    )

    st.info(
        recommendations.get(
            prediction,
            "No recommendation available."
        )
    )

    pdf_file = create_pdf_report(

        current_date,

        prediction,

        best_confidence,

        health,
        risk,

 

        recommendations.get(
            prediction,
            ""
        ),
        original_image_path,
        fault_image_path
    )

    st.download_button(

        label="📄 Download PDF Report",

        data=pdf_file,

        file_name="Thermal_Report.pdf",

        mime="application/pdf"
    )
