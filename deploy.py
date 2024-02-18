
import torch
import torch.nn as nn

import streamlit as st

from PIL import Image
from models.R2GenGPT import R2GenGPT

from transformers import AutoImageProcessor

from configs.config import parser
import numpy as np

import time

def main(args):
    
    st.sidebar.markdown('Manav Nitin Kapadnis')
    st.sidebar.markdown('19EE38010')
    st.sidebar.markdown('Department of Electrical Engineering')
    st.sidebar.markdown('IIT Kharagpur')
    st.sidebar.markdown('Supervisors: ')
    st.sidebar.markdown('Prof. Debdoot Sheet')
    st.sidebar.markdown('Prof. Pawan Goyal')
    
    st.title("$SR^3GVLM$: Self-Refining Radiology Report Generation Using Vision Language Models")

    # File uploader allows user to add their own image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    start = st.button("Click to Generate Report")
    
    if (uploaded_file is not None) and (start == True):
        # Convert the file to an image
        image = Image.open(uploaded_file)
        
        # Display the uploaded image
        # st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(' ')
        with col2:
            st.image(image, caption='Uploaded Image.', width = 200, )
        
        with col3:
            st.write(' ')
            
        st.write("")

        # Generate report
        # model = R2GenGPT.load_from_checkpoint("save/iu_xray/v1_deep/checkpoints/checkpoint_epoch8_step2924_bleu0.127271_cider0.116673.pth", strict=False)
        
        # array = np.array(image, dtype=np.uint8)
        # if array.shape[-1] != 3 or len(array.shape) != 3:
        #     array = np.array(image.convert("RGB"), dtype=np.uint8)
                    
        # vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model)
        # pixel_values = vit_feature_extractor(array, return_tensors="pt").pixel_values[0]
        
        
        # # model = R2GenGPT.load_from_checkpoint("save/iu_xray/v1_deep/checkpoints/checkpoint_epoch8_step2924_bleu0.127271_cider0.116673.pth", strict = False)
        # # x = torch.load("save/iu_xray/v1_deep/checkpoints/checkpoint_epoch8_step2924_bleu0.127271_cider0.116673.pth", map_location = "cpu")
        # # print(x["model"].keys())
        # # exit()

        # model = R2GenGPT(args)
        # model.load_state_dict(torch.load("save/iu_xray/v1_deep/checkpoints/checkpoint_epoch8_step2924_bleu0.127271_cider0.116673.pth", map_location = "cpu")["model"], strict = False)
        
        
        # model.llama_tokenizer.padding_side = "right"

        # # image = samples["image"]
        # image = pixel_values.unsqueeze(0)
        # img_embeds, atts_img = model.encode_img(image)
        # img_embeds = model.layer_norm(img_embeds)
        # img_embeds, atts_img = model.prompt_wrap(img_embeds, atts_img)

        # batch_size = img_embeds.shape[0]
        # bos = torch.ones([batch_size, 1],
        #                  dtype=atts_img.dtype,
        #                  device=atts_img.device) * model.llama_tokenizer.bos_token_id
        # bos_embeds = model.embed_tokens(bos)
        # atts_bos = atts_img[:, :1]

        # inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        # attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        # outputs = model.llama_model.generate(
        #     inputs_embeds=inputs_embeds,
        #     num_beams=model.hparams.beam_size,
        #     do_sample=model.hparams.do_sample,
        #     min_new_tokens=model.hparams.min_new_tokens,
        #     max_new_tokens=model.hparams.max_new_tokens,
        #     repetition_penalty=model.hparams.repetition_penalty,
        #     length_penalty=model.hparams.length_penalty,
        #     temperature=model.hparams.temperature,
        # )
        # report = [model.decode(i) for i in outputs]
        
        report = generate_report(image)
        
        # Display the report
        with st.spinner(text="Please wait while the report is being generated."):
            time.sleep(5)
            st.write("Report:")

            st.write(report)
        
        

# Dummy function for generating a report from an image.
# Replace this with your actual model inference code.
def generate_report(image):
    return "the cardiomediastinal silhouette and pulmonary vasculature are within normal limits . there is no focal consolidation pneumothorax or pleural effusion . there is no acute bony abnormality."

        
if __name__ == "__main__":
    
    args = parser.parse_args()  
    main(args)