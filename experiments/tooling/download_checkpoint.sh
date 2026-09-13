#!/usr/bin/env bash
# Public checkpoint, exact immutable revision; local experiment environment only.
set -euo pipefail
ROOT=/opt/robopad/checkpoints/smolvla_libero
REVISION=6721902bc4d61e50a3bfdb11dfb4cb626f05d102
mkdir -p "$ROOT"
for FILE in README.md config.json model.safetensors policy_postprocessor.json policy_postprocessor_step_1_unnormalizer_processor.safetensors policy_preprocessor.json policy_preprocessor_step_5_normalizer_processor.safetensors; do
  if [ ! -f "$ROOT/$FILE" ]; then
    if [ "$FILE" = model.safetensors ]; then
      aria2c --max-connection-per-server=16 --split=16 --min-split-size=1M \
        --file-allocation=none --summary-interval=60 --console-log-level=warn \
        --show-console-readout=false --continue=true --max-tries=10 --retry-wait=3 \
        --checksum=sha-256=71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f \
        --dir="$ROOT" --out=model.parallel.safetensors \
        "https://huggingface.co/HuggingFaceVLA/smolvla_libero/resolve/$REVISION/$FILE"
      mv "$ROOT/model.parallel.safetensors" "$ROOT/$FILE"
    else
      curl --fail --location --retry 8 --retry-all-errors --retry-delay 3 --continue-at - \
        --silent --show-error \
        "https://huggingface.co/HuggingFaceVLA/smolvla_libero/resolve/$REVISION/$FILE" \
        --output "$ROOT/$FILE.part"
      mv "$ROOT/$FILE.part" "$ROOT/$FILE"
    fi
  fi
done
cd "$ROOT"
printf '%s\n' \
  '71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f  model.safetensors' \
  '7008ba73c73887e5a7e631d478de9563ec5a0c51b7f0327f2530d7326c4b039e  policy_postprocessor_step_1_unnormalizer_processor.safetensors' \
  '7008ba73c73887e5a7e631d478de9563ec5a0c51b7f0327f2530d7326c4b039e  policy_preprocessor_step_5_normalizer_processor.safetensors' | sha256sum --check
printf '%s\n' "$REVISION" > "$ROOT/.robopad_revision"
sha256sum -- *.json *.safetensors > download.sha256
cat download.sha256
