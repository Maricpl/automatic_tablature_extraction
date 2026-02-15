import gradio as gr
from tablature_extraction.source_separation import SeparationHub
from tablature_extraction.transcription import TrascriptionHub
from tablature_extraction.pipeline import TablatureGenerationPipeline
from scipy.io.wavfile import write
import numpy as np

def inference(input_audio, separation_model, transcription_model):
    # create tmp wav, as models use paths instead of samples
    sr, y = input_audio
    tmp_wav = "tmp.wav"
    write(tmp_wav, sr, y)

    pipeline = TablatureGenerationPipeline(
        separation_model=separation_model,
        transcription_model=transcription_model,
    )
    guitar_path, midi_data = pipeline.inference(tmp_wav)
    original_spectrogram = pipeline.source_separation.plot_spectrogram(tmp_wav)
    guitar_spectrogram = pipeline.source_separation.plot_spectrogram(guitar_path)
    midi_obj = midi_data["midi_data"]
    piano_roll = pipeline.transcription.plot_piano_roll(midi_obj)
    synthezized_transcription = midi_obj.synthesize(wave=np.sin)
    # If you want to display MIDI as text, you can add pretty_midi or similar here
    tab_text = midi_data["tabs_content"]
    return guitar_path, original_spectrogram, guitar_spectrogram, piano_roll, (44100, synthezized_transcription), tab_text


css="""
#output-tabs-box textarea {
    font-family: 'Fira Mono', 'Consolas', 'Menlo', 'Monaco', 'Liberation Mono', monospace;
    font-size: 15px;
    white-space: pre;
}
"""

with gr.Blocks(css=css) as demo:
    separation_model = gr.Dropdown(
        label="Separation Model",
        choices=SeparationHub.get_available_models(),
        value="dttnet",
    )
    transcription_model = gr.Dropdown(
        label="Transcription Model",
        choices=TrascriptionHub.get_available_models(),
        value="crnn",
    )
    # Removed tablature_transcription_model UI
    input_audio = gr.Audio(label="Input Audio")
    
    inference_btn = gr.Button("Inference")
    
    output_audio = gr.Audio(label='output_audio')
    original_spectrogram = gr.Plot(label="Original spectrogram")
    output_spectrogram = gr.Plot(label="Result spectrogram for guitar stem")
    transcription_piano_roll = gr.Plot(label="Transcription Piano Roll")
    synthezized_transcription = gr.Audio(label="Synthezized Transcription")
    output_tabs = gr.Textbox(label="Output Tablature", placeholder="Generated tablature will appear here...", lines=12, elem_id="output-tabs-box")

    inference_btn.click(
        fn=inference,
        inputs=[input_audio, separation_model, transcription_model],
        outputs=[output_audio, original_spectrogram, output_spectrogram, transcription_piano_roll, synthezized_transcription, output_tabs],
        api_name="Inference"
    )
    
# formating output tabs with monospace font for better readability
demo.launch(share=True)