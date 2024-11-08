1. In text to ISL
Setup this server first
https://stanfordnlp.github.io/CoreNLP/download.html
Download this dataset
https://www.kaggle.com/datasets/koushikchouhan/indian-sign-language-animated-videos
to start a server:
1) cd C:\Users\stanford-corenlp-4.5.7 //change your path
2) java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

to run py file
python convert2isl.py "college finish"

2. for Speech to text download model from this link
save this at Speech_to_text/model/ and also at model/
https://vitedu-my.sharepoint.com/:u:/g/personal/yatharth_thakare211_vit_edu/EQdecCa5EzxFhHTcqaFffFYB1aeN3e6Vq9ixL7HGAh-Fgg?e=yucS76


Current code works. 
WORK ON TEXT_TO_SPEECH


