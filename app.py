import gradio as gr
from inference import predict



demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath"),
    outputs="text",
)

if __name__ == "__main__":
    demo.launch(share=True)
