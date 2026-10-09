# Pretrained model

[sshleifer/distilbart-cnn-12-6](https://huggingface.co/sshleifer/distilbart-cnn-12-6)
is a 306M-parameter English news summarizer with 12 encoder and 6 decoder layers.
Its checkpoint is already fine-tuned on CNN/DailyMail. This project performs
inference only. Its repository is labeled Apache-2.0.

`model_manifest.json` records the exact upstream revision and local checksums.
`distilbart-cnn-12-6/` holds downloaded tokenizer/config files and pretrained
weights. These large files are ignored by Git and recreated by preparation.
Inference uses only local files, so the API never downloads weights on a request.
