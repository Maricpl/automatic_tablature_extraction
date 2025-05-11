import gradio as gr
from tablature_extraction.source_separation import SeparationHub
from tablature_extraction.transcription import TrascriptionHub
from tablature_extraction.pipeline import TablatureGenerationPipeline

def inference(separation_model, transcription_model):
    pipeline = TablatureGenerationPipeline(
        separation_model=separation_model,
        transcription_model=transcription_model,
    )
    res = pipeline.inference_mock("data/songs/mettalica_10s.wav")
    return res

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
    output = gr.Plot(label="Result spectrogram")
    input_audio = gr.Audio(label="Input Audio")
    inference_btn = gr.Button("Inference")
    
    inference_btn.click(fn=inference, inputs=[separation_model, transcription_model], outputs=output, api_name="Inference")

demo.launch(share=True)