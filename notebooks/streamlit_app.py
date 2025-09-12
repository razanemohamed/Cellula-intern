import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# ----------------------
# 🔹 Background Styling
# ----------------------
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         [data-testid="stAppViewContainer"] {{
             background-color: lightblue;
             background-image: url("https://www.google.com/imgres?q=teeth%20clinic%20photo&imgurl=https%3A%2F%2Fwww.shutterstock.com%2Fimage-photo%2Fdental-hygiene-oral-health-care-600nw-2523738153.jpg&imgrefurl=https%3A%2F%2Fwww.shutterstock.com%2Fsearch%2Fclinic-teeth&docid=Vv1xtf9DSeyQ5M&tbnid=fun3oSERvNwVhM&vet=12ahUKEwijlNyPwdOPAxXjdqQEHazIOfMQM3oECBcQAA..i&w=600&h=300&hcb=2&ved=2ahUKEwijlNyPwdOPAxXjdqQEHazIOfMQM3oECBcQAA");
             background-size: cover;
             background-position: center;
             background-attachment: fixed;
         }}

         [data-testid="stHeader"] {{
             background: rgba(0,0,0,0);  /* transparent header */
         }}

         [data-testid="stSidebar"] {{
             background-color: rgba(255, 255, 255, 0.7);
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

# ----------------------
# 🔹 Cached Model Loader
# ----------------------
@st.cache_resource
def load_model():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Class names
    class_names = ["CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT"]

    # Load pretrained EfficientNet and adjust for 7 classes
    model = models.efficientnet_b0(weights=None)  # no pretrained weights
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))

    # Load your trained weights
    state_dict = torch.load("efficientnet_b0_best.pth", map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()

    return model, DEVICE, class_names


# ----------------------
# 🔹 Transforms
# ----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.47365978360176086, 0.5072661638259888, 0.4893795847892761], [0.2911527752876282, 0.29588305950164795, 0.2806512713432312])
])


# ----------------------
# 🔹 Streamlit UI
# ----------------------
st.title("🦷 Teeth Classification Demo (EfficientNet-B0)")
st.markdown("Upload a teeth image to classify into 7 categories")

uploaded_file = st.file_uploader("Upload a teeth image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load model only once (cached)
    model, DEVICE, class_names = load_model()

    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()
        pred_class = class_names[pred_idx]
        confidence = probs[pred_idx].item()

    st.markdown(f"### ✅ Prediction: **{pred_class}** ({confidence*100:.2f}%)")

    # Show all class probabilities as a bar chart
    st.bar_chart({cls: float(probs[i]) for i, cls in enumerate(class_names)})
