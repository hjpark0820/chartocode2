# Related Work and References

A curated list of open-source projects, tools, and research papers relevant to chart data extraction from scientific literature.

## Libraries Used in This Project

| Project | Role in Our Pipeline | Link |
|---------|---------------------|------|
| **Ultralytics YOLOv8** | Symbol localization (object detection). We use the nano variant (~3M params) | [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) |
| **SAHI** | Sliced Aided Hyper Inference — improves recall for small symbols by running detection on overlapping image tiles | [github.com/obss/sahi](https://github.com/obss/sahi) |
| **scikit-learn** | RANSAC outlier filtering for removing false-positive data points | [github.com/scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) |
| **PyTorch** | Framework for our custom CNN classifier (94K params, 9-class) | [github.com/pytorch/pytorch](https://github.com/pytorch/pytorch) |

## Scatter Plot / Chart Data Extraction (Traditional CV)

These projects tackle the same core problem using classical computer vision and/or conventional deep learning:

| Project | Description | Link |
|---------|-------------|------|
| **Scatteract** (Bloomberg) | Automatic data extraction from scatter plot images. Uses TensorBox for detecting points and tick marks + Tesseract for OCR. Closest in approach to our pipeline | [github.com/bloomberg/scatteract](https://github.com/bloomberg/scatteract) |
| **ChartReader / ChartOCR** | End-to-end framework using VGG/ResNet/EfficientNet for chart type classification + AWS Rekognition for text detection | [github.com/Cvrane/ChartReader](https://github.com/Cvrane/ChartReader) |
| **GraphMaster** | Fully automated framework for extracting data from complex scientific charts using pixel projection methods | [github.com/MasterAI-EAM/GraphMaster](https://github.com/MasterAI-EAM/GraphMaster) |
| **Plot2Spec** | Automatic plot digitizer for spectroscopy images (XANES, Raman). Uses an anchor-free detector with edge-based refinement | [github.com/MaterialEyes/Plot2Spec](https://github.com/MaterialEyes/Plot2Spec) |
| **LineFormer** (ICDAR 2023) | Line chart data extraction using instance segmentation. Published at ICDAR 2023 with academic paper | [github.com/TheJaeLal/LineFormer](https://github.com/TheJaeLal/LineFormer) |
| **Information Extraction from Scientific Data Charts** | Image processing application for automatic data point extraction from scientific charts | [github.com/arpitjainds/Information-Extraction-from-Scientific-Data-Charts](https://github.com/arpitjainds/Information-Extraction-from-Scientific-Data-Charts) |

## Chart Understanding with Large Models (Multimodal / LLM-Based)

Emerging approaches that use foundation models or multimodal LLMs for chart understanding:

| Project | Description | Link |
|---------|-------------|------|
| **DePlot** (Google Research) | Chart-to-table modality conversion model based on Pix2Struct. Translates a chart image directly into a linearized data table, which can then be reasoned over by an LLM | [github.com/google-research/google-research (deplot)](https://github.com/google-research/google-research/tree/master/deplot) |
| **MatCha** (Google Research) | Pixels-to-text foundation model trained on chart de-rendering and math reasoning tasks. Related to DePlot | [HuggingFace: google/matcha-base](https://huggingface.co/google/matcha-base) |
| **ChartLlama** | Multimodal LLM for chart understanding and generation, built on LLaVA-1.5. Outperforms prior methods on ChartQA and chart-to-text benchmarks | [github.com/tingxueronghua/ChartLlama-code](https://github.com/tingxueronghua/ChartLlama-code) |
| **MMC** (NAACL 2024) | Multimodal Chart understanding with LLM instruction tuning | [github.com/FuxiaoLiu/MMC](https://github.com/FuxiaoLiu/MMC) |

## Survey Papers and Curated Lists

| Resource | Description | Link |
|----------|-------------|------|
| **Awesome-Chart-Understanding** | Curated list of chart understanding work, companion to an IEEE TKDE survey paper | [github.com/khuangaf/Awesome-Chart-Understanding](https://github.com/khuangaf/Awesome-Chart-Understanding) |
| **Awesome-LLM-for-Chart** | Curated list of papers on using LLMs for chart data extraction, visual reasoning, and chart-based QA | [github.com/rongzizi/Awesome-LLM-for-Chart](https://github.com/rongzizi/Awesome-LLM-for-Chart) |

## Semi-Automatic Digitizer Tools

Interactive tools where users manually calibrate axes and the tool assists with point extraction:

| Tool | Description | Link |
|------|-------------|------|
| **WebPlotDigitizer** | The most mature web-based chart digitizer. Semi-automatic with manual axis calibration. Widely used in academic research | [automeris.io](https://automeris.io/) |
| **PinPoint Digitizer** | Open-source cross-platform desktop application. Supports both linear and logarithmic axes | [github.com/mhismail/PinPoint-Digitizer](https://github.com/mhismail/PinPoint-Digitizer) |
| **PlotDigitizer** | Python CLI utility for digitizing plots, supports batch processing | [github.com/dilawar/PlotDigitizer](https://github.com/dilawar/PlotDigitizer) |
| **StarryDigitizer** | Open-source web-based tool with multiple automated extraction modes | [digitizer.starrydata.org](https://digitizer.starrydata.org/) |
