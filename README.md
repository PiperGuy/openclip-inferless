# Open CLIP in Inferless.com
`Inferless.com` is a cloud platform with serverless GPU execution model in which the  provider allocates machine resources on demand, taking care of the servers on behalf of their customers.

`Open CLIP` is an open source implementation of OpenAI's CLIP (Contrastive Language-Image Pre-training) model used recognize a wide variety of visual concepts in images and associate them with their names. 

You can use this repo to import the model in Inferless.

---
## Prerequisites
- **Git**. You would need git installed on your system if you wish to customize the repo after cloning/forking.
- **Python>=3.8**. You would need Python to customize the code in the `app.py` according to your needs.
- **Curl**. You would need Curl if you want to make API calls from the terminal itself.

### File structure requirements
The below format should be kept in mind while loading your code to Inferless from GitHub.
1. The mandatory requirement would be an **`app.py`** file.
2. This file should contain the below functions(Mandatory)
	1. `def initialize(self)`
	2. `def infer(self, inputs)`
	3. `def finalize(self)`

## Quick Start
Here is a quick start to help you get up and running with this template on Inferless.

### Clone the Repository
Get started by cloning the repository. You can do this by clicking on the Code button in the top right corner of the repository page.

This will create a copy of the repository in your local system, allowing you to make changes and customize it according to your needs.

### Fork the Repository
Get started by forking the repository. You can do this by clicking on the fork button in the top right corner of the repository page.

This will create a copy of the repository in your own GitHub account, allowing you to make changes and customize it according to your needs.

### Create a Custom Runtime in Inferless
To make inferences with OpenCLIP model we need to create a custom **runtime** to install [torch](https://pytorch.org/get-started/locally/#linux-python), [pillow](https://pypi.org/project/Pillow/) and [open-clip-torch](https://pypi.org/project/open-clip-torch/).

To access the custom runtime window in Inferless, simply navigate to the sidebar and click on the **Create new Runtime** button. A pop-up will appear.

Next, provide a suitable name (i.e. **openclip**) for your custom runtime and proceed by uploading the `runtime/openclip.yaml` file given above. Finally, ensure you save your changes by clicking on the save button.

### Import the Model in Inferless
Log in to your inferless account, select the workspace you want the model to be imported into and click the Add Model button.

Select the PyTorch as framework and choose **Repo(custom code)** as your model source and use the repo URL as the **Model URL**.

After the create model step, while setting the configuration for the model make sure to select the appropriate runtime.

Enter all the required details to Import your model. Refer [this link](https://docs.inferless.com/integrations/github-custom-code) for more information on model import.

The following is a sample Input and Output JSON for this model which you can use while importing this model on Inferless.

### Input
```json
{
  "inputs": [
    {
      "name": "text",
      "shape": [-1],
      "datatype": "BYTES",
      "data": ["red sofa"]
    }
  ]
}
```

### Output
```json
{
  "outputs": [
    {
      "name": "embeddings",
      "shape": [-1,-1],
      "datatype": "FP32",
      "data": [
        [
        -0.07819648087024689,
        1.3333008289337158,
        -0.10979261249303818,
        -0.27275943756103516
        ]
      ]
    },
	{
	  "name": "model",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["OpenCLIP-ViT-B-32-laion2B"]
    },
	{
	  "name": "duration",
      "shape": [1],
      "datatype": "FP16",
      "data": ["2.0"]
    }
  ]
}
```

---
## Curl Command
Following is an example of the curl command you can use to make inference. You can find the exact curl command in the Model's API page in Inferless.

```bash
curl --location '<inference_url>' \
          --header 'Content-Type: application/json' \
          --header 'Authorization: Bearer <api_key>' \
          --data '{
					"inputs": [
						{
						"name": "text",
						"shape": [
							1
						],
						"datatype": "BYTES",
						"data": [
							"red sofa"
						]
						}
					]
				}'
```

---
## Customizing the Code
Open the `app.py` file. This contains the main code for inference. It has three main functions, initialize, infer and finalize.

**Initialize** -  This function is executed during the cold start and is used to initialize the model. If you have any custom configurations or settings that need to be applied during the initialization, make sure to add them in this function.

```python
def initialize(self):
	self.model, _, self.preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=MODEL_PRETRAINED, cache_dir=CACHE_DIR, device=self.device)
```

**Infer** - This function is where the inference happens. The argument to this function `inputs`, is a dictionary containing all the input parameters. The keys are the same as the name given in inputs. Refer to [input](#input) for more.

```python
def infer(self, inputs):
	text = self.tokenizer(inputs.get("text")).to(self.device)
	text_embeddings = self.model.encode_text(text).tolist()
```

**Finalize** - This function is used to perform any cleanup activity for example you can unload the model from the gpu by setting `self.pipe = None`.


For more information refer to the [Inferless docs](https://docs.inferless.com/).