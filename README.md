## Dataset

We use the Mozilla Common Voice dataset (Korean and Russian subsets).

- Website: https://commonvoice.mozilla.org
- License: CC0 1.0 Universal (Public Domain)

Common Voice is a crowdsourced multilingual speech dataset released by Mozilla.

Although CC0 does not require attribution, we acknowledge and thank Mozilla and the contributors for providing this dataset.

## Experiments
- dataset: commonvoice ko 100 + ru 100
- model: whisper-small
- preprocessing: librosa 16kHz, text normalization
- result: preprocessing success

## Run Pipeline

This script runs the full inference pipeline:

```text
Audio Input → Whisper + LoRA ASR → Transcription → Feedback Output
```

### 1. Prepare sample audio

Place an audio file in:

```text
data/samples/
```

Example:

```text
data/samples/test.mp3
```

### 2. Run with automatic language detection

```bash
python src/run_pipeline.py data/samples/test.mp3 --language auto
```

### 3. Run with Korean mode

```bash
python src/run_pipeline.py data/samples/ko_test.mp3 --language ko
```

### 4. Run with Russian mode

```bash
python src/run_pipeline.py data/samples/ru_test.mp3 --language ru
```

### 5. Run with code-switching audio

```bash
python src/run_pipeline.py data/samples/codeswitch_test.mp3 --language auto
```

### Expected Output

```text
========== PIPELINE RESULT ==========

[AUDIO]
data/samples/test.mp3

[TRANSCRIPTION]
дробь에서 분모는 아래에 있는 수입니다

[FEEDBACK]
코드스위칭 감지: 러시아어 키릴 문자가 포함되어 있습니다.
дробь: 러시아어로 '분수'
분모: 분수에서 아래에 있는 수

=====================================
```

### Notes

- Use `--language auto` for code-switching audio.
- Use `--language ko` for Korean-only audio.
- Use `--language ru` for Russian-only audio.
- The pipeline uses the LoRA adapter from:

```text
outputs/whisper_lora_final/final
```

If the final model does not exist, it can fall back to:

```text
outputs/whisper_lora_manual/final
```

## Batch Pipeline Test

여러 오디오 파일에 대해 전체 파이프라인을 자동으로 실행하고 결과를 저장합니다.

```text
Audio files → run_pipeline.py → Transcription + Feedback → JSON result
```

### 1. 테스트 오디오 준비

테스트 오디오는 아래 폴더에 넣습니다.

```text
data/samples/
```

예시:

```text
data/samples/ko_test.mp3
data/samples/ru_test.mp3
data/samples/codeswitch_test.mp3
```

### 2. Batch 테스트 실행

```bash
python src/test_pipeline_batch.py
```

### 3. 결과 분석 실행

```bash
python src/analyze_pipeline_results.py
```

### 4. 생성 결과

```text
results/pipeline_batch_outputs.json
results/pipeline_batch_summary.txt
```

### 5. 결과 예시

```text
========== Pipeline Batch Summary ==========
Total files : 3
Success     : 3
Error       : 0
============================================

총 3개의 테스트 오디오에 대해 파이프라인을 실행한 결과,
모든 샘플에서 음성 인식 및 피드백 생성이 정상적으로 수행되었다.
```
