from config import MODEL_NAME, MODEL_PRETRAINED, MODEL_ID, CACHE_DIR
import torch
# from PIL import Image
import open_clip
import time



class InferlessPythonModel:

    # Implement the Load function here for the model
	def initialize(self):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model, _, self.preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=MODEL_PRETRAINED, cache_dir=CACHE_DIR, device=self.device)
		self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)


	# Function to perform inference 
	def infer(self, inputs):
		# inputs is a dictonary where the keys are input names and values are actual input data
		# e.g. in the below code the input name is "prompt"
		# prompt = inputs["prompt"]

		if (inputs.get("text") == None):
			raise ValueError("the parameter 'Inputs' must contain 'text' or 'image' value") 

		start_time = time.perf_counter()



		text = self.tokenizer(inputs.get("text")).to(self.device)
		text_embeddings = self.model.encode_text(text).tolist()

		image_embeddings = []
		return {	
			"model": MODEL_ID,
			"inputs" : inputs,
			"embeddings": text_embeddings + image_embeddings,
			"duration": time.perf_counter() - start_time
			}

	# perform any cleanup activity here
	def finalize(self, args):
		self.device, self.model, self.preprocess, self.tokenizer = None


# o = InferlessPythonModel()
# o.initialize()
# result = o.infer({"text":"red sofa"})
# print(result)
