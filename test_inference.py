import pytest
import base64
from app import *



@pytest.fixture
def o():
	o = InferlessPythonModel()
	o.initialize()
	return o


def test_infer_text(o):
	result = o.infer({"text": "sofa"})
	print(result)
	assert len(result.get("embeddings")) > 0


def test_infer_image(o):
	with open("a_641175_2.jpg", "rb") as f:
		im_bytes = f.read()        
	im_b64 = base64.b64encode(im_bytes).decode("utf8")
	result = o.infer({"image": im_b64})
	print(result)
	assert len(result.get("embeddings")) > 0