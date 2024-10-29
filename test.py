from transformers import pipeline, AutoFeatureExtractor,AutoModelForAudioClassification
#這兩行改路徑 save_model的路徑
loaded_model = AutoModelForAudioClassification.from_pretrained("C:/Users/JiaXun/Desktop/tt/Transformer/save_model")
loaded_feature_extractor = AutoFeatureExtractor.from_pretrained("C:/Users/JiaXun/Desktop/tt/Transformer/save_model")
pipe = pipeline("audio-classification", model=loaded_model,
				feature_extractor=loaded_feature_extractor,device=0)

import pandas as pd
from datasets import Dataset




def classify_audio(filepath):
	preds = pipe(filepath)
	outputs = {}
	for p in preds:
		outputs[p["label"]] = p["score"]
	return outputs


        
 
    
 
    
 
input_file_path = input("input file:")    
 
    
path_new=input_file_path.replace("\\","/")

output = classify_audio(path_new)


print("Predicted Genre:")
max_key = max(output, key=output.get)

print("The predicted genre is:", max_key)
print("The prediction score is:", output[max_key])