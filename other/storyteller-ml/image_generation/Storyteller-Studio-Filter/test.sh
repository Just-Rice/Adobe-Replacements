python3.10 main.py \
--prompt "raiden mei, pantyhose, boots, single glove, earrings, dress, choker, tank top" \
--negative-prompt "This is a negative prompt" \
--number-of-samples 128 \
--samplers "DPM++ 2M SDE Heun Karras" \
--width 1024 \
--height 1024 \
--cfg-scale 8.5 \
--seed 1 \
--loRA-path "xs4vow.safetensors" \
--check-point "waifuDiffusionBeta03_beta3.safetensors" \
--vae "vaeFtEma560000Ema_original.pt" \
--batch-size 1 \
--batch-count 10

