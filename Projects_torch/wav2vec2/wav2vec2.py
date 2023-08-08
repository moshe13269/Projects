import torch
import torchaudio
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torchaudio.utils import download_asset

from scipy.io import wavfile



bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model()


p = r"C:\Users\moshe\PycharmProjects\commercial_synth_dataset\noy\data\10002.wav"
_, data = wavfile.read(p)
data = torch.unsqueeze(torch.from_numpy(data), dim=0)


with torch.inference_mode():
    features, _ = model.extract_features(data) #torch.normal(mean=torch.zeros(1, 16834)))

# fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
# for i, feats in enumerate(features):
#     ax[i].imshow(feats[0].cpu(), interpolation="nearest")
#     ax[i].set_title(f"Feature from transformer layer {i+1}")
#     ax[i].set_xlabel("Feature dimension")
#     ax[i].set_ylabel("Frame (time-axis)")
# plt.tight_layout()
# plt.show()

with torch.inference_mode():
    emission, _ = model(data)
    emission1, _ = model(data1)
    emission2, _ = model(data2)

e = emission[0].T.numpy()
e1 = emission1[0].T.numpy()
e2 = emission2[0].T.numpy()

plt.imshow(emission[0].cpu().T, interpolation="nearest")
plt.title("Classification result")
plt.xlabel("Frame (time-axis)")
plt.ylabel("Class")
plt.show()
print("Class labels:", bundle.get_labels())