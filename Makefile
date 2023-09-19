installtransformers:
	installtransformers1 installtransformers2 installtransformers3

installtransformers1:
	@git clone https://github.com/huggingface/transformers.git                             [🐍 moder_ia]

installtransformers2:
	@cd transformers

installtransformers3:
	@pip install .

install:
	@pip install -e .
