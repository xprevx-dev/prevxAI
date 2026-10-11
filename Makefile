.PHONY: install test smoke data train sft chat serve clean help

PRESET ?= rtx3050_4gb
DATA   ?= data
OUT    ?= runs/chatbot

help:
	@echo "make install   install PyTorch (CPU wheel; see requirements.txt for CUDA)"
	@echo "make test      run the test suite"
	@echo "make smoke     end-to-end proof on a tiny corpus (runs on a CPU, ~1 min)"
	@echo "make data      download TinyShakespeare and tokenise it"
	@echo "make train     pretrain (PRESET=$(PRESET))"
	@echo "make sft       instruction fine-tune into a chatbot"
	@echo "make chat      chat in the terminal"
	@echo "make serve     serve the web UI + OpenAI-compatible API"

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -q

data:
	python scripts/prepare_data.py --tinyshakespeare --out-dir $(DATA)

train:
	python -m myai.train --preset $(PRESET) --data-dir $(DATA) --out-dir $(OUT)

sft:
	python -m myai.sft --checkpoint $(OUT)/best.pt --data $(DATA)/chat.jsonl --out-dir $(OUT)

chat:
	python -m myai.generate --checkpoint $(OUT)/sft.pt --chat

serve:
	python -m myai.serve --checkpoint $(OUT)/sft.pt --port 8000

# The full pipeline on a small corpus, small enough for a CPU. Proves every
# stage works before you spend GPU hours.
smoke:
	python scripts/prepare_data.py --tinyshakespeare --num-merges 1000 --out-dir data
	python -m myai.train --preset smoke --data-dir data --out-dir runs/smoke \
		--max-steps 300 --batch-size 8 --block-size 128 --n-embd 128 --n-head 4 --n-layer 3 \
		--eval-interval 150 --save-interval 150
	python -m myai.sft --checkpoint runs/smoke/best.pt --data data/sample_chat.jsonl --max-steps 100
	python -m myai.generate --checkpoint runs/smoke/sft.pt --prompt "ROMEO:\n" --max-new-tokens 50

clean:
	rm -rf runs data *.egg-info .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
