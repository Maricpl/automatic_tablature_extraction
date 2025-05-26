import gradio as gr
from tablature_extraction.source_separation import SeparationHub
from tablature_extraction.transcription import TrascriptionHub, TablatureTrascriptionHub
from tablature_extraction.pipeline import TablatureGenerationPipeline
from scipy.io.wavfile import write
import numpy as np

def inference(input_audio, separation_model, transcription_model, tablature_transcription_model):
    # create tmp wav, as models use paths instead of samples
    sr, y = input_audio
    tmp_wav = "tmp.wav"
    write(tmp_wav, sr, y)

    pipeline = TablatureGenerationPipeline(
        separation_model=separation_model,
        transcription_model=transcription_model,
        tablature_transcription_model=tablature_transcription_model,
    )
    guitar_path, midi_data, tabs = pipeline.inference(tmp_wav)
    original_spectrogram = pipeline.source_separation.plot_spectrogram(tmp_wav)
    guitar_spectrogram = pipeline.source_separation.plot_spectrogram(guitar_path)
    
    piano_roll = pipeline.transcription.plot_piano_roll(midi_data)

    synthezized_transcription = midi_data.synthesize(wave = np.sin)

    return guitar_path, original_spectrogram, guitar_spectrogram, piano_roll, (44100, synthezized_transcription), tabs

with gr.Blocks() as demo:
    separation_model = gr.Dropdown(
        label="Separation Model",
        choices=SeparationHub.get_available_models(),
        value="open_unmix",
    )
    transcription_model = gr.Dropdown(
        label="Transcription Model",
        choices=TrascriptionHub.get_available_models(),
        value="basic_pitch",
    )
    tablature_transcription_model = gr.Dropdown(
        label="Tablature Transcription Model",
        choices=TablatureTrascriptionHub.get_available_models(),
        value="tayuya",
    )
    input_audio = gr.Audio(label="Input Audio")
    
    inference_btn = gr.Button("Inference")
    
    output_audio = gr.Audio(label='output_audio')
    original_spectrogram = gr.Plot(label="Original spectrogram")
    output_spectrogram = gr.Plot(label="Result spectrogram for guitar stem")
    transcription_piano_roll = gr.Plot(label="Transcription Piano Roll")
    synthezized_transcription = gr.Audio(label="Synthezized Transcription")
    output_tabs = gr.Textbox(label="Output Tablature", placeholder="Generated tablature will appear here...")

    inference_btn.click(fn=inference, 
                        inputs=[input_audio, separation_model, transcription_model, tablature_transcription_model], 
                        outputs=[output_audio, original_spectrogram, output_spectrogram, transcription_piano_roll, synthezized_transcription, output_tabs], 
                        api_name="Inference")
    

demo.launch(share=True)