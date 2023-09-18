installtransformers:
	@git clone https://github.com/huggingface/transformers.git                             [🐍 moder_ia]
	@cd transformers
	@pip install .

install:
	@pip install -e .
