import onnxruntime as ort
import numpy as np
from PIL import Image

session = ort.InferenceSession('animal_model.onnx')
input_name = session.get_inputs()[0].name
CLASSES = ['cat', 'dog', 'humans', 'wild']

def predict(img_path):
    image = Image.open(img_path).convert('RGB').resize((224, 224))
    img = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0).astype(np.float32)
    logits = session.run(None, {input_name: img})[0][0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    idx = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx])

cls, conf = predict('test_cat.png')
print(f"test_cat.png -> predicted: '{cls}' ({conf*100:.1f}% confidence)")
print(f"is_human: {cls == 'humans'}")
