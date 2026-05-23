"""
Gradio web interface for cotton leaf disease prediction.

Usage:
    python app/interface.py

Expected input:
    models/best_model.pth

Python 3.10+, Gradio 4.0+
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# -----------------------------------------------------------------------------
# Central configuration
#
# Paths and preprocessing constants mirror the training/evaluation scripts so
# the UI uses the same model and image normalization contract.
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_model.pth"
IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# -----------------------------------------------------------------------------
# Uzbek display labels and disease descriptions
#
# The checkpoint keeps English folder-style class names. The UI maps those raw
# class keys to Uzbek labels for display while preserving the model output order.
# -----------------------------------------------------------------------------
CLASS_LABELS_UZ = {
    "bacterial_blight": "Bakterial Dog'",
    "curl_virus": "Barg Buralishi",
    "fusarium_wilt": "Fusarium So'lishi",
    "healthy": "Sog'lom Barg",
}

DISEASE_DESCRIPTIONS = {
    "bacterial_blight": """🦠 Bakterial Dog' (Bacterial Blight)

Sabab: Xanthomonas citri pv. malvacearum bakteriyasi
Belgilar:
  • Bargda suv singgan, yog'li ko'rinishdagi dog'lar
  • Dog'lar keyinchalik qo'ng'ir-qora rangga o'tadi
  • Barg chetlari qurib, to'kiladi
  • Shoxlarda qorayish kuzatiladi

Davolash:
  • Mis asosidagi fungitsidlar (Bordeaux aralashmasi)
  • Kasallangan barglarni olib yoqish
  • Ekin almashlab ekish
  • Chidamli navlarni ekish

Zarar darajasi: ⚠️ Yuqori — hosil 30-40% kamayishi mumkin""",
    "curl_virus": """🌀 Barg Buralishi (Cotton Leaf Curl Virus)

Sabab: Cotton leaf curl virus (CLCuV) — oq kapalakcha orqali tarqaladi
Belgilar:
  • Barglar yuqoriga yoki pastga buraladi
  • Barg tomirlarida qalinlashish
  • O'simlik o'sishdan to'xtaydi (stunting)
  • Barglar qadoqlanib, qattiq tortadi

Davolash:
  • Oq kapalakchaga qarshi insektitsidlar
  • Kasallangan o'simliklarni darhol yo'q qilish
  • Virusga chidamli nav ekish (CLCuV-resistant)
  • Begona o't nazorati

Zarar darajasi: 🔴 Juda yuqori — epidemiya xavfi bor""",
    "fusarium_wilt": """🍂 Fusarium So'lishi (Fusarium Wilt)

Sabab: Fusarium oxysporum f. sp. vasinfectum zamburug'i
Belgilar:
  • Barglar sarg'ayib, bir tomondan so'liydi
  • O'simlik ildizi va poyasida qo'ng'ir rang
  • Poya kesimida qoraygan tomirlar ko'rinadi
  • O'simlik asta-sekin qurib o'ladi

Davolash:
  • Tuproqni fungitsid bilan ishlov berish
  • Kasallangan ildizlarni yo'q qilish
  • 4-5 yil ekin almashlab ekish
  • Trichoderma asosidagi biologik preparatlar

Zarar darajasi: ⚠️ O'rta-Yuqori — tuproqda 10 yil saqlanadi""",
    "healthy": """✅ Sog'lom Barg

Holat: O'simlik sog'lom, kasallik belgilari yo'q

Sog'lom o'simlik belgilari:
  • Barg rangi — to'q yashil, bir xil
  • Barg yuzasi — silliq, dog'siz
  • O'simlik o'sishi — me'yorda
  • Tomirlar — aniq, yashil

Profilaktika tavsiyalari:
  • Har 2 haftada barglarni tekshirib turing
  • Sug'orishni me'yorida bering
  • Mineral o'g'itlarni me'yorida qo'llang
  • Dalani begona o'tlardan tozalab turing""",
}


# -----------------------------------------------------------------------------
# Example images
#
# Gradio displays these local test images under the upload component so users
# can try the interface immediately.
# -----------------------------------------------------------------------------
EXAMPLES = [
    [str(PROJECT_ROOT / "data" / "processed" / "test" / "bacterial_blight" / "20241029_132352.jpg")],
    [str(PROJECT_ROOT / "data" / "processed" / "test" / "curl_virus" / "20241029_135328.jpg")],
    [str(PROJECT_ROOT / "data" / "processed" / "test" / "fusarium_wilt" / "20241029_135119.jpg")],
]


# -----------------------------------------------------------------------------
# Model and transform setup
#
# The checkpoint stores class_names. Loading them from the checkpoint prevents
# the probability dictionary from drifting away from the model output order.
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def load_model() -> tuple[nn.Module, list[str]]:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    class_names = checkpoint["class_names"]

    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features=1280, out_features=len(class_names)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names


MODEL, CLASS_NAMES = load_model()


# -----------------------------------------------------------------------------
# Prediction function
#
# Gradio passes a PIL image. The function returns Uzbek-labeled probabilities
# for gr.Label and a detailed description for gr.Textbox.
# -----------------------------------------------------------------------------
def predict(image: Image.Image | None) -> tuple[dict[str, float], str]:
    if image is None:
        return {}, "Iltimos, paxta bargi rasmini yuklang."

    image_tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(image_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    raw_probability_dict = {
        class_name: float(probability)
        for class_name, probability in zip(CLASS_NAMES, probabilities.tolist())
    }
    probability_dict = {
        CLASS_LABELS_UZ.get(class_name, class_name): probability
        for class_name, probability in raw_probability_dict.items()
    }
    top_class = max(raw_probability_dict, key=raw_probability_dict.get)
    description = DISEASE_DESCRIPTIONS.get(top_class, "Tavsif topilmadi.")

    return probability_dict, description


# -----------------------------------------------------------------------------
# Gradio interface
#
# The interface provides a drag-and-drop image input, a four-class probability
# output, a disease-description textbox, and three sample images.
# -----------------------------------------------------------------------------
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Drag & drop rasm yuklash"),
    outputs=[
        gr.Label(num_top_classes=4, label="Barcha sinflar foizi"),
        gr.Textbox(label="Kasallik tavsifi", lines=18),
    ],
    title="🌿 Paxta Bargi Kasallik Aniqlagich",
    description="Paxta bargi rasmini drag & drop orqali yuklang.",
    examples=EXAMPLES,
)


if __name__ == "__main__":
    interface.launch(share=True)
