---
license: apache-2.0
license_link: https://www.apache.org/licenses/LICENSE-2.0
pipeline_tag: image-text-to-text
tags:
- conversational
- image-text-to-text
- moe
library_name: transformers
---

# TML-d24-varloop-research

<img src="https://cdn-uploads.huggingface.co/production/uploads/630e8f0bf6f6d700f50ebd2e/AvmDwmrWRMnKjOWvmLieg.png" style="display: block;margin-left: auto;margin-right: auto;width: 30%;">

## 1. General Information

TML-d24-varloop-research is a multimodal research model that accepts text and image inputs and generates text outputs. It is a variable-looped counterpart to Inkling 2 Mini: whereas Inkling 2 Mini always runs its looped layers a fixed number of times, the number of loops in this model can be chosen at inference time. We are releasing it with open weights to support research on looped models. Because the number of loops can be varied on a single set of weights, the model allows cleaner comparisons of how additional looped computation affects a model's internal computation and overall capabilities.

**Languages:** English, general multilingual support

**Model provider:** Thinking Machines Lab, Inc.

**Date of release:** Nov 20, 2026

**License:** Apache 2.0

[Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/)

## 2. Getting Started

The weights are available for download through Hugging Face [here](https://huggingface.co/thinkingmachines/tml-d24-varloop-research).

The model is not available through an API, including Tinker. It is distributed as open weights only.

### Required hardware

The model is distributed as a single checkpoint of about 20GB. Its expert layers are NVFP4, the last four use MXFP8, and the remaining weights are BF16.

The checkpoint requires a single machine with at least 32GB of GPU or unified memory. This can be met with any of the following example configurations:

* 1x NVIDIA DGX Spark
* 1x NVIDIA GPU with 32GB or more of VRAM (for example, GeForce RTX 5090 or RTX PRO 6000 Blackwell)
* Apple silicon Mac with 32GB or more of unified memory (for example, Mac mini)

The model uses the same mixed NVFP4 and MXFP8 precision it was trained in, so this checkpoint is its native precision and needs no further quantization.

### Required software

TML-d24-varloop-research supports local inference using the following frameworks, along with their dependency libraries:

* [Hugging Face](https://hf.co/blog/inkling2mini)
* [SGLang](https://docs.sglang.io/cookbook/autoregressive/ThinkingMachines/Inkling-2-Mini)
* [vLLM](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini)
* [TokenSpeed](https://lightseek.org/tokenspeed/recipes/models#inkling-2-mini)
* NVIDIA Dynamo
* [Unsloth](https://unsloth.ai/docs/inkling-2-mini), including quantized GGUF versions
* llama.cpp, for quantized GGUF versions
* [Llama App](https://llama.app/)
* Ollama and Exo Labs, on Apple silicon

TML-d24-varloop-research can also be post-trained with:

* [Miles](https://miles.radixark.com/docs/models/thinkingmachines/inkling-2-mini)
* [SkyRL](https://docs.skyrl.ai/docs/models/inkling2)

## 3. Model Properties

### Model type

Multimodal autoregressive looped transformer

### Architecture type

A 24-layer decoder-only transformer with sparse Mixture-of-Experts (MoE) feed-forward layers. The middle 12 layers run a variable number of times with shared weights, chosen at inference time; the model was trained with 1 to 5 passes. In each layer, each token is routed to 8 of 128 experts. Attention is a hybrid of local and global layers. The model is natively multimodal – images are encoded via a vision transformer – with image and text inputs projected into a shared hidden space and processed jointly by the decoder.

### Parameters

26.4B, 3.4B active

Note: each extra pass through the looped layers adds about 1.0B effective active parameters per token. This corresponds to 3.4B active parameters at 1 pass, 5.4B at 3 passes, and 7.4B at 5 passes.

### Numerics support

NVFP4 and MXFP8

### Input modalities

* Text: UTF-8 encoded text
* Image: Any pixel-based image input. For optimal performance, each image dimension should be between 40px to 4096px.

TML-d24-varloop-research supports a context window of up to 1M tokens.

### Output modalities

Text

## 4. Training

Training data includes a broad variety of content types, including text, images, video. Training data for the model was drawn from publicly available sources, acquired from third-parties, generated internally, or synthetically generated or augmented. Publicly available data includes content from the public internet and publicly accessible repositories.

Data curation includes cleaning, processing, and modifying datasets. These processing steps, which vary by data type, may include deduplication and filtering to remove junk or other low-quality data, or to advance safety or other objectives.

### Training with variable looping

TML-d24-varloop-research shares its pretraining and first stage of midtraining with Inkling 2 Mini. For the rest of midtraining, it was trained with 1 to 5 passes through the looped layers, sampled in a 1:2:6:2:1 ratio, and it was then post-trained with 3 passes. Because 3 passes dominated training once looping was introduced, performance at other pass counts may understate what models trained specifically at those counts could achieve. The two models' later training stages differ, so we caution against comparing TML-d24-varloop-research directly to Inkling 2 Mini.

## 5. Evaluations

Each row shows one benchmark score for TML-d24-varloop-research run at 1 to 5 passes through its looped layers, beside Inkling 2 Mini, which always runs 3 passes. Scores are in percent, except PMPP-Hard, an area under the curve of tasks solved over 0–10 hours.

<table>
<thead>
<tr><th style="border-right-color:rgba(128,128,128,0.45)"></th><th style="font-weight:500;border-right-color:rgba(128,128,128,0.45)">Inkling 2 Mini</th><th colspan="5" style="font-weight:500">TML-d24-varloop-research</th></tr>
<tr><th style="border-right-color:rgba(128,128,128,0.45)"></th><th style="font-weight:400;border-right-color:rgba(128,128,128,0.45)">(3 passes)</th><th style="font-weight:400">1 pass</th><th style="font-weight:400">2 passes</th><th style="font-weight:400">3 passes</th><th style="font-weight:400">4 passes</th><th style="font-weight:400">5 passes</th></tr>
</thead>
<tbody>
<tr><td style="background-color:rgba(128,128,128,0.1);font-weight:600;border-right-color:rgba(128,128,128,0.45)">Agentic Coding</td><td style="background-color:rgba(128,128,128,0.1);border-right-color:rgba(128,128,128,0.45)"></td><td style="background-color:rgba(128,128,128,0.1)"></td><td style="background-color:rgba(128,128,128,0.1)"></td><td style="background-color:rgba(128,128,128,0.1)"></td><td style="background-color:rgba(128,128,128,0.1)"></td><td style="background-color:rgba(128,128,128,0.1)"></td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">DeepSWE v1.1<br><span style="font-size:11px;opacity:0.7">mini-swe-agent v2.4.6</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">48.2%</td><td style="text-align:center;vertical-align:middle">0.7%</td><td style="text-align:center;vertical-align:middle">28.3%</td><td style="text-align:center;vertical-align:middle">47.4%</td><td style="text-align:center;vertical-align:middle">46.9%</td><td style="text-align:center;vertical-align:middle">48.2%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">SWEBench Pro v2 (Hard)<br><span style="font-size:11px;opacity:0.7">mini-swe-agent v2.4.6</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">77.9%</td><td style="text-align:center;vertical-align:middle">39.7%</td><td style="text-align:center;vertical-align:middle">69.1%</td><td style="text-align:center;vertical-align:middle">78.9%</td><td style="text-align:center;vertical-align:middle">76.0%</td><td style="text-align:center;vertical-align:middle">78.4%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">SWEBench Multilingual<br><span style="font-size:11px;opacity:0.7">bash only</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">73.7%</td><td style="text-align:center;vertical-align:middle">40.8%</td><td style="text-align:center;vertical-align:middle">68.5%</td><td style="text-align:center;vertical-align:middle">71.8%</td><td style="text-align:center;vertical-align:middle">72.8%</td><td style="text-align:center;vertical-align:middle">72.0%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">SWEBench Verified<br><span style="font-size:11px;opacity:0.7">bash only</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">79.2%</td><td style="text-align:center;vertical-align:middle">56.3%</td><td style="text-align:center;vertical-align:middle">76.5%</td><td style="text-align:center;vertical-align:middle">78.7%</td><td style="text-align:center;vertical-align:middle">79.3%</td><td style="text-align:center;vertical-align:middle">79.6%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">SWEBench Multimodal v2<br><span style="font-size:11px;opacity:0.7">mini-swe-agent (vals)</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">37.9%</td><td style="text-align:center;vertical-align:middle">23.1%</td><td style="text-align:center;vertical-align:middle">32.9%</td><td style="text-align:center;vertical-align:middle">34.8%</td><td style="text-align:center;vertical-align:middle">35.4%</td><td style="text-align:center;vertical-align:middle">36.0%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">PMPP-Hard (AUC, 0–10h)<br><span style="font-size:11px;opacity:0.7">pi v0.83.0</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">455.8</td><td style="text-align:center;vertical-align:middle">49.5</td><td style="text-align:center;vertical-align:middle">315.3</td><td style="text-align:center;vertical-align:middle">401.8</td><td style="text-align:center;vertical-align:middle">437.2</td><td style="text-align:center;vertical-align:middle">424.0</td></tr>
<tr><td style="background-color:rgba(128,128,128,0.1);font-weight:600;border-right-color:rgba(128,128,128,0.45)">Agentic Search, Tool Use, and Knowledge Work</td><td style="background-color:rgba(128,128,128,0.1);border-right-color:rgba(128,128,128,0.45)"></td><td style="background-color:rgba(128,128,128,0.1)"></td><td style="background-color:rgba(128,128,128,0.1)"></td><td style="background-color:rgba(128,128,128,0.1)"></td><td style="background-color:rgba(128,128,128,0.1)"></td><td style="background-color:rgba(128,128,128,0.1)"></td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">BrowseComp<br><span style="font-size:11px;opacity:0.7">original, no context management</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">76.0%</td><td style="text-align:center;vertical-align:middle">29.4%</td><td style="text-align:center;vertical-align:middle">53.8%</td><td style="text-align:center;vertical-align:middle">65.7%</td><td style="text-align:center;vertical-align:middle">69.4%</td><td style="text-align:center;vertical-align:middle">71.0%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">DeepSearchQA<br><span style="font-size:11px;opacity:0.7">with compaction</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">85.3%</td><td style="text-align:center;vertical-align:middle">58.5%</td><td style="text-align:center;vertical-align:middle">74.8%</td><td style="text-align:center;vertical-align:middle">78.9%</td><td style="text-align:center;vertical-align:middle">83.4%</td><td style="text-align:center;vertical-align:middle">83.0%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">MCP Atlas<br><span style="font-size:11px;opacity:0.7">Public</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">82.7%</td><td style="text-align:center;vertical-align:middle">70.0%</td><td style="text-align:center;vertical-align:middle">77.6%</td><td style="text-align:center;vertical-align:middle">79.4%</td><td style="text-align:center;vertical-align:middle">79.7%</td><td style="text-align:center;vertical-align:middle">80.0%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45)">JobBench</td><td style="text-align:center;border-right-color:rgba(128,128,128,0.45)">49.6%</td><td style="text-align:center">22.8%</td><td style="text-align:center">55.1%</td><td style="text-align:center">63.8%</td><td style="text-align:center">61.8%</td><td style="text-align:center">63.3%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">EnterpriseOps-Gym<br><span style="font-size:11px;opacity:0.7">LangChain ReAct loop</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">36.7%</td><td style="text-align:center;vertical-align:middle">25.7%</td><td style="text-align:center;vertical-align:middle">34.4%</td><td style="text-align:center;vertical-align:middle">36.6%</td><td style="text-align:center;vertical-align:middle">36.5%</td><td style="text-align:center;vertical-align:middle">37.6%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">Diligence Bench<br><span style="font-size:11px;opacity:0.7">Loop</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">31.7%</td><td style="text-align:center;vertical-align:middle">16.1%</td><td style="text-align:center;vertical-align:middle">22.2%</td><td style="text-align:center;vertical-align:middle">27.9%</td><td style="text-align:center;vertical-align:middle">29.4%</td><td style="text-align:center;vertical-align:middle">32.3%</td></tr>
<tr><td style="background-color:rgba(128,128,128,0.1);font-weight:600;border-right-color:rgba(128,128,128,0.45)">Vision Interpretation and Creation</td><td style="background-color:rgba(128,128,128,0.1);border-right-color:rgba(128,128,128,0.45)"></td><td style="background-color:rgba(128,128,128,0.1)"></td><td style="background-color:rgba(128,128,128,0.1)"></td><td style="background-color:rgba(128,128,128,0.1)"></td><td style="background-color:rgba(128,128,128,0.1)"></td><td style="background-color:rgba(128,128,128,0.1)"></td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45)">GDP.pdf</td><td style="text-align:center;border-right-color:rgba(128,128,128,0.45)">16.8%</td><td style="text-align:center">4.6%</td><td style="text-align:center">11.2%</td><td style="text-align:center">19.6%</td><td style="text-align:center">18.0%</td><td style="text-align:center">20.0%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45)">PerceptionBench</td><td style="text-align:center;border-right-color:rgba(128,128,128,0.45)">57.5%</td><td style="text-align:center">45.5%</td><td style="text-align:center">51.3%</td><td style="text-align:center">53.9%</td><td style="text-align:center">53.7%</td><td style="text-align:center">52.7%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">MMMU Pro<br><span style="font-size:11px;opacity:0.7">Standard 10</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">74.9%</td><td style="text-align:center;vertical-align:middle">63.6%</td><td style="text-align:center;vertical-align:middle">69.8%</td><td style="text-align:center;vertical-align:middle">71.4%</td><td style="text-align:center;vertical-align:middle">72.1%</td><td style="text-align:center;vertical-align:middle">73.2%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45)">OmniDocBench v1.6</td><td style="text-align:center;border-right-color:rgba(128,128,128,0.45)">92.8%</td><td style="text-align:center">86.6%</td><td style="text-align:center">90.5%</td><td style="text-align:center">92.1%</td><td style="text-align:center">91.8%</td><td style="text-align:center">92.5%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">BabyVision<br><span style="font-size:11px;opacity:0.7">original</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">53.4%</td><td style="text-align:center;vertical-align:middle">32.7%</td><td style="text-align:center;vertical-align:middle">44.9%</td><td style="text-align:center;vertical-align:middle">50.3%</td><td style="text-align:center;vertical-align:middle">50.3%</td><td style="text-align:center;vertical-align:middle">49.2%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">BabyVision<br><span style="font-size:11px;opacity:0.7">with Python</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">88.8%</td><td style="text-align:center;vertical-align:middle">67.2%</td><td style="text-align:center;vertical-align:middle">84.1%</td><td style="text-align:center;vertical-align:middle">88.9%</td><td style="text-align:center;vertical-align:middle">88.8%</td><td style="text-align:center;vertical-align:middle">89.5%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">ScreenSpot Pro<br><span style="font-size:11px;opacity:0.7">original</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">76.5%</td><td style="text-align:center;vertical-align:middle">54.4%</td><td style="text-align:center;vertical-align:middle">66.0%</td><td style="text-align:center;vertical-align:middle">72.0%</td><td style="text-align:center;vertical-align:middle">73.4%</td><td style="text-align:center;vertical-align:middle">73.1%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">ScreenSpot Pro<br><span style="font-size:11px;opacity:0.7">with Python</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">87.7%</td><td style="text-align:center;vertical-align:middle">80.7%</td><td style="text-align:center;vertical-align:middle">86.5%</td><td style="text-align:center;vertical-align:middle">87.7%</td><td style="text-align:center;vertical-align:middle">87.2%</td><td style="text-align:center;vertical-align:middle">88.1%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">Charxiv RQ<br><span style="font-size:11px;opacity:0.7">original</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">80.6%</td><td style="text-align:center;vertical-align:middle">75.0%</td><td style="text-align:center;vertical-align:middle">78.3%</td><td style="text-align:center;vertical-align:middle">79.9%</td><td style="text-align:center;vertical-align:middle">79.2%</td><td style="text-align:center;vertical-align:middle">80.4%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">Charxiv RQ<br><span style="font-size:11px;opacity:0.7">with Python</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">88.3%</td><td style="text-align:center;vertical-align:middle">84.7%</td><td style="text-align:center;vertical-align:middle">87.7%</td><td style="text-align:center;vertical-align:middle">87.6%</td><td style="text-align:center;vertical-align:middle">89.3%</td><td style="text-align:center;vertical-align:middle">87.9%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">Chartography<br><span style="font-size:11px;opacity:0.7">original</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">22.1%</td><td style="text-align:center;vertical-align:middle">14.5%</td><td style="text-align:center;vertical-align:middle">17.8%</td><td style="text-align:center;vertical-align:middle">21.6%</td><td style="text-align:center;vertical-align:middle">21.4%</td><td style="text-align:center;vertical-align:middle">22.6%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">Chartography<br><span style="font-size:11px;opacity:0.7">with Python</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">72.5%</td><td style="text-align:center;vertical-align:middle">49.1%</td><td style="text-align:center;vertical-align:middle">69.1%</td><td style="text-align:center;vertical-align:middle">72.3%</td><td style="text-align:center;vertical-align:middle">73.7%</td><td style="text-align:center;vertical-align:middle">72.3%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">ZeroBench (main)<br><span style="font-size:11px;opacity:0.7">original</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">17.0%</td><td style="text-align:center;vertical-align:middle">9.0%</td><td style="text-align:center;vertical-align:middle">13.0%</td><td style="text-align:center;vertical-align:middle">19.0%</td><td style="text-align:center;vertical-align:middle">18.0%</td><td style="text-align:center;vertical-align:middle">17.0%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">ZeroBench (main)<br><span style="font-size:11px;opacity:0.7">with Python</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">34.0%</td><td style="text-align:center;vertical-align:middle">16.0%</td><td style="text-align:center;vertical-align:middle">31.0%</td><td style="text-align:center;vertical-align:middle">40.0%</td><td style="text-align:center;vertical-align:middle">35.0%</td><td style="text-align:center;vertical-align:middle">37.0%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45)">Humanity’s Sixth Sense</td><td style="text-align:center;border-right-color:rgba(128,128,128,0.45)">23.8%</td><td style="text-align:center">22.4%</td><td style="text-align:center">24.8%</td><td style="text-align:center">25.5%</td><td style="text-align:center">26.5%</td><td style="text-align:center">27.0%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45)">MathVision</td><td style="text-align:center;border-right-color:rgba(128,128,128,0.45)">89.0%</td><td style="text-align:center">74.8%</td><td style="text-align:center">84.9%</td><td style="text-align:center">86.6%</td><td style="text-align:center">87.8%</td><td style="text-align:center">88.1%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45)">MuirBench</td><td style="text-align:center;border-right-color:rgba(128,128,128,0.45)">77.5%</td><td style="text-align:center">73.6%</td><td style="text-align:center">76.1%</td><td style="text-align:center">77.1%</td><td style="text-align:center">76.4%</td><td style="text-align:center">77.8%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45)">BVB</td><td style="text-align:center;border-right-color:rgba(128,128,128,0.45)">68.4%</td><td style="text-align:center">58.1%</td><td style="text-align:center">64.4%</td><td style="text-align:center">68.5%</td><td style="text-align:center">69.2%</td><td style="text-align:center">69.5%</td></tr>
<tr><td style="background-color:rgba(128,128,128,0.1);font-weight:600;border-right-color:rgba(128,128,128,0.45)">Reasoning, Knowledge, and Epistemics</td><td style="background-color:rgba(128,128,128,0.1);border-right-color:rgba(128,128,128,0.45)"></td><td style="background-color:rgba(128,128,128,0.1)"></td><td style="background-color:rgba(128,128,128,0.1)"></td><td style="background-color:rgba(128,128,128,0.1)"></td><td style="background-color:rgba(128,128,128,0.1)"></td><td style="background-color:rgba(128,128,128,0.1)"></td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45)">HLE-Diamond</td><td style="text-align:center;border-right-color:rgba(128,128,128,0.45)">15.2%</td><td style="text-align:center">6.3%</td><td style="text-align:center">8.5%</td><td style="text-align:center">9.5%</td><td style="text-align:center">9.7%</td><td style="text-align:center">11.1%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45)">HLE (text only)</td><td style="text-align:center;border-right-color:rgba(128,128,128,0.45)">30.7%</td><td style="text-align:center">10.5%</td><td style="text-align:center">20.0%</td><td style="text-align:center">24.9%</td><td style="text-align:center">25.4%</td><td style="text-align:center">25.9%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45)">GPQA Diamond</td><td style="text-align:center;border-right-color:rgba(128,128,128,0.45)">86.6%</td><td style="text-align:center">69.6%</td><td style="text-align:center">79.2%</td><td style="text-align:center">84.2%</td><td style="text-align:center">81.9%</td><td style="text-align:center">83.1%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45)">Apex 2025</td><td style="text-align:center;border-right-color:rgba(128,128,128,0.45)">34.4%</td><td style="text-align:center">3.1%</td><td style="text-align:center">10.4%</td><td style="text-align:center">19.8%</td><td style="text-align:center">15.6%</td><td style="text-align:center">18.8%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">ComplexConstraints v1.1<br><span style="font-size:11px;opacity:0.7">openhands v1.45.0</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">33.4%</td><td style="text-align:center;vertical-align:middle">4.5%</td><td style="text-align:center;vertical-align:middle">26.0%</td><td style="text-align:center;vertical-align:middle">37.9%</td><td style="text-align:center;vertical-align:middle">39.6%</td><td style="text-align:center;vertical-align:middle">38.9%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45)">SimpleQA Verified</td><td style="text-align:center;border-right-color:rgba(128,128,128,0.45)">8.4%</td><td style="text-align:center">6.3%</td><td style="text-align:center">6.5%</td><td style="text-align:center">7.3%</td><td style="text-align:center">6.8%</td><td style="text-align:center">6.7%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">AA-Omniscience v1.1 (Public)<br><span style="font-size:11px;opacity:0.7">accuracy</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">21.3%</td><td style="text-align:center;vertical-align:middle">15.5%</td><td style="text-align:center;vertical-align:middle">16.3%</td><td style="text-align:center;vertical-align:middle">18.4%</td><td style="text-align:center;vertical-align:middle">18.0%</td><td style="text-align:center;vertical-align:middle">17.4%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">AA-Omniscience v1.1 (Public)<br><span style="font-size:11px;opacity:0.7">non-hallucination rate</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">64.6%</td><td style="text-align:center;vertical-align:middle">79.3%</td><td style="text-align:center;vertical-align:middle">85.0%</td><td style="text-align:center;vertical-align:middle">84.0%</td><td style="text-align:center;vertical-align:middle">85.7%</td><td style="text-align:center;vertical-align:middle">85.9%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45)">SimpleVQA</td><td style="text-align:center;border-right-color:rgba(128,128,128,0.45)">45.4%</td><td style="text-align:center">39.0%</td><td style="text-align:center">41.8%</td><td style="text-align:center">42.4%</td><td style="text-align:center">42.9%</td><td style="text-align:center">41.3%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45)">WorldVQA</td><td style="text-align:center;border-right-color:rgba(128,128,128,0.45)">15.2%</td><td style="text-align:center">11.3%</td><td style="text-align:center">13.0%</td><td style="text-align:center">14.2%</td><td style="text-align:center">14.3%</td><td style="text-align:center">13.5%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45)">AA-LCR v1.1</td><td style="text-align:center;border-right-color:rgba(128,128,128,0.45)">80.0%</td><td style="text-align:center">55.5%</td><td style="text-align:center">73.5%</td><td style="text-align:center">74.8%</td><td style="text-align:center">75.5%</td><td style="text-align:center">76.8%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45)">LongBench v2</td><td style="text-align:center;border-right-color:rgba(128,128,128,0.45)">66.9%</td><td style="text-align:center">50.6%</td><td style="text-align:center">61.8%</td><td style="text-align:center">66.5%</td><td style="text-align:center">66.2%</td><td style="text-align:center">66.4%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">GraphWalks BFS<br><span style="font-size:11px;opacity:0.7">256K</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">69.7%</td><td style="text-align:center;vertical-align:middle">21.1%</td><td style="text-align:center;vertical-align:middle">54.5%</td><td style="text-align:center;vertical-align:middle">63.2%</td><td style="text-align:center;vertical-align:middle">64.3%</td><td style="text-align:center;vertical-align:middle">64.9%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">GraphWalks BFS<br><span style="font-size:11px;opacity:0.7">1M</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">50.0%</td><td style="text-align:center;vertical-align:middle">5.7%</td><td style="text-align:center;vertical-align:middle">39.4%</td><td style="text-align:center;vertical-align:middle">52.1%</td><td style="text-align:center;vertical-align:middle">56.2%</td><td style="text-align:center;vertical-align:middle">57.8%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">MRCR v2<br><span style="font-size:11px;opacity:0.7">8-needle 256K–512K</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">46.4%</td><td style="text-align:center;vertical-align:middle">26.2%</td><td style="text-align:center;vertical-align:middle">65.2%</td><td style="text-align:center;vertical-align:middle">89.2%</td><td style="text-align:center;vertical-align:middle">93.4%</td><td style="text-align:center;vertical-align:middle">92.1%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">MRCR v2<br><span style="font-size:11px;opacity:0.7">8-needle 512K–1M</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">37.7%</td><td style="text-align:center;vertical-align:middle">20.6%</td><td style="text-align:center;vertical-align:middle">49.7%</td><td style="text-align:center;vertical-align:middle">72.3%</td><td style="text-align:center;vertical-align:middle">81.4%</td><td style="text-align:center;vertical-align:middle">79.9%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">ARC-AGI-1<br><span style="font-size:11px;opacity:0.7">Public</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">90.5%</td><td style="text-align:center;vertical-align:middle">29.5%</td><td style="text-align:center;vertical-align:middle">59.3%</td><td style="text-align:center;vertical-align:middle">71.5%</td><td style="text-align:center;vertical-align:middle">72.6%</td><td style="text-align:center;vertical-align:middle">75.3%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">ARC-AGI-2<br><span style="font-size:11px;opacity:0.7">Public</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">51.3%</td><td style="text-align:center;vertical-align:middle">0.8%</td><td style="text-align:center;vertical-align:middle">12.2%</td><td style="text-align:center;vertical-align:middle">22.1%</td><td style="text-align:center;vertical-align:middle">20.3%</td><td style="text-align:center;vertical-align:middle">27.1%</td></tr>
<tr><td style="background-color:rgba(128,128,128,0.1);font-weight:600;border-right-color:rgba(128,128,128,0.45)">Safety and Alignment</td><td style="background-color:rgba(128,128,128,0.1);border-right-color:rgba(128,128,128,0.45)"></td><td style="background-color:rgba(128,128,128,0.1)"></td><td style="background-color:rgba(128,128,128,0.1)"></td><td style="background-color:rgba(128,128,128,0.1)"></td><td style="background-color:rgba(128,128,128,0.1)"></td><td style="background-color:rgba(128,128,128,0.1)"></td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">FORTRESS (adversarial)<br><span style="font-size:11px;opacity:0.7">Public</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">89.5%</td><td style="text-align:center;vertical-align:middle">89.0%</td><td style="text-align:center;vertical-align:middle">90.4%</td><td style="text-align:center;vertical-align:middle">90.0%</td><td style="text-align:center;vertical-align:middle">89.9%</td><td style="text-align:center;vertical-align:middle">89.4%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45);vertical-align:middle">FORTRESS (benign)<br><span style="font-size:11px;opacity:0.7">Public</span></td><td style="text-align:center;vertical-align:middle;border-right-color:rgba(128,128,128,0.45)">96.4%</td><td style="text-align:center;vertical-align:middle">95.0%</td><td style="text-align:center;vertical-align:middle">95.9%</td><td style="text-align:center;vertical-align:middle">95.3%</td><td style="text-align:center;vertical-align:middle">95.7%</td><td style="text-align:center;vertical-align:middle">95.0%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45)">StrongREJECT (none)</td><td style="text-align:center;border-right-color:rgba(128,128,128,0.45)">97.8%</td><td style="text-align:center">98.8%</td><td style="text-align:center">98.4%</td><td style="text-align:center">98.2%</td><td style="text-align:center">99.4%</td><td style="text-align:center">98.4%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45)">PatchEval</td><td style="text-align:center;border-right-color:rgba(128,128,128,0.45)">60.9%</td><td style="text-align:center">55.7%</td><td style="text-align:center">73.9%</td><td style="text-align:center">73.9%</td><td style="text-align:center">75.2%</td><td style="text-align:center">73.5%</td></tr>
<tr style="white-space:nowrap"><td style="border-right-color:rgba(128,128,128,0.45)">Destructive actions (Internal) ↓</td><td style="text-align:center;border-right-color:rgba(128,128,128,0.45)">10.2%</td><td style="text-align:center">26.6%</td><td style="text-align:center">16.4%</td><td style="text-align:center">7.8%</td><td style="text-align:center">3.1%</td><td style="text-align:center">7.0%</td></tr>
</tbody>
</table>


## 6. Safety

We evaluated this model with the same pre-release process as Inkling 2 Mini. It is less capable than models whose weights we have already released on CBRN and cyber evaluations, and it refuses harmful requests about as reliably as Inkling 2 Small.

For production use, we recommend Inkling 2 Mini as it generally performs better on capabilities and alignment evaluations. We do not intend TML-d24-varloop-research for production use.

Across all areas, we concluded that TML-d24-varloop-research did not present risk of material uplift beyond what’s already available in the open-weight ecosystem.

## 7. Bias, risks and limitations

TML-d24-varloop-research is a research model, not a production model. It may exhibit general limitations common to foundation models, including hallucination (generating plausible but factually incorrect or unsupported content), occasional failures to follow instructions precisely, and degraded performance in long multi-turn conversations. As with other large-scale models trained on web-derived and synthetic data, TML-d24-varloop-research may reflect biases present in its training data, including demographic, cultural, or linguistic biases, and may perform unevenly across languages, dialects, or subject domains that were less represented during training.

TML-d24-varloop-research’s knowledge is limited to information available as of its training cutoff, and it may not reflect events, developments, or changes that occurred afterward.

TML-d24-varloop-research is intended to be a research artifact. It has not been carefully tuned for deployment, and may behave unpredictably at pass counts other than 3. We do not recommend deploying this model in production or user-facing applications.

## 8. Legal

[Training Data Documentation](https://thinkingmachines.ai/training-data-documentation/)
