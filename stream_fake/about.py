import streamlit as st
from PIL import Image
import io


def show_about():
    st.header(' ')
    st.header(' ')
    st.header(' ')
    #st.header("API")
    st.write("A deepfake is a type of synthetic media generated using deep learning techniques, particularly deep neural networks. The term ""deepfake"" is a combination of ""deep learning"" and ""fake.""")
    st.write("In deepfake technology, algorithms are used to create or manipulate audio,\
             video, or images to depict something that did not actually occur or that alters the appearance or actions of individuals.\
             This can involve superimposing images or videos of people onto existing footage,\
             making individuals appear to say or do things they never said or did.")
    st.header("Deepfake Fraud Statistics")
    #st.write("")
    image = Image.open("C:\\Users\\Paras Sharma\\OneDrive\\Pictures\\Saved Pictures\\deepfake-growth.jpg")
    new_image = image.resize((600, 400))
    st.image(new_image)

    
    st.header("Real Cases:")
    st.markdown("**Kerala Man Loses Rs 40,000 to AI-Based Deepfake WhatsApp Fraud**")
    st.write("According to India Today, a man in Kerala lost Rs 40,000 in an online scam on WhatsApp involving\
             AI-based deepfake technology. The scammer impersonated the victim's former\
             colleague via video call, fabricating a medical emergency and requesting\
             money. This incident underscores the danger of sophisticated online fraud\
             using deepfake technology and emphasizes the importance of verifying\
             unexpected financial requests to avoid falling victim to such scams.")
    



    '''st.header("The Dangers of Deepfakes")
    st.write("Organizations and individuals are at risk regarding deepfakes as it’s a source that leverages social engineering attempts to manufacture fraudulent texts, voice messages, and fake videos to spread misinformation. \
    According to the US Department of Defense, deepfakes are AI-generated, highly realistic content that can be used to: \
    Threaten an organization’s brand. \
    Impersonate leaders and financial officers. \
    Enable access to networks, communications, and other sensitive information. \
    In this sense, all companies that are housing business and customer data could be at risk to these attacks.")
'''
    
