# import torch
# import torchaudio
# from speechbrain.pretrained import SpectralMaskEnhancement
#
# enhance_model = SpectralMaskEnhancement.from_hparams(
#     source="speechbrain/metricgan-plus-voicebank",
#     savedir="pretrained_models/metricgan-plus-voicebank",
# )
#
# # Load and add fake batch dimension
# noisy = enhance_model.load_audio(
#     "speechbrain/metricgan-plus-voicebank/example.wav"
# ).unsqueeze(0)
#
# # Add relative length tensor
# enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]))
#
# # Saving enhanced signal on disk
# torchaudio.save('enhanced.wav', enhanced.cpu(), 16000)


from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

inputs = processor(text="Hello, my dog is cute", return_tensors="pt")

# load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

sf.write("speech.wav", speech.numpy(), samplerate=16000)