from transformers import pipeline
import gradio

get_completion = pipeline("summarization",model="sshleifer/distilbart-cnn-12-6")

def summarize(input):
    output = get_completion(input)
    return output[0]['summary_text']

gradio.close_all()
# demo = gradio.Interface(fn=summarize,inputs="text", outputs="text")
demo = gradio.Interface(fn=summarize,inputs=[gradio.Textbox(label="Text to summarize",lines=7)], 
                        outputs=[gradio.Textbox(label="Result",lines=3)],
                        title="Text Summarization",
                        description="Text Summarization using the sshleifer/distilbart-cnn-12-6 model")
demo.launch(share=True)
