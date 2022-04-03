# GPT3 + Beam Search + Sampling

За основу взято решение для BART c [huggingface](https://github.com/huggingface/transformers/tree/main/examples/research_projects/onnx/summarization)

## Описание

Код конвертирует модель из Pytorch в Onnx с поддержкой генерации текста:

* beam search;
* temperature;
* top p.  

## Пример запуска

```bash
python run_onnx_exporter.py --model_checkpoint_path savedir/s_kw/checkpoints/ --output_file_path ./savedir/s_kw/gpt3.onnx --model_tokenizer_path savedir/tokenizer --num_beams 4 --max_length 20 --temperature 1.0 --top_p 0.95 --device cpu
```
