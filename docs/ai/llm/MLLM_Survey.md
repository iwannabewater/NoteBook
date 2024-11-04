# MLLM (Multi-modal Large Language Model)

## 1. LLAVA:

**GitHub repo link:**

https://github.com/haotian-liu/LLaVA

**Project Page:**

https://llava-vl.github.io/

**Demo:**

https://llava.hliu.cc/

### 1.1 About:

**[NeurIPS 2023 Oral]** Visual Instruction Tuning: LLaVA (Large Language-and-Vision Assistant) built towards GPT-4V level capabilities.

![image-20231027174819999](C:\Users\KevinGeorge\AppData\Roaming\Typora\typora-user-images\image-20231027174819999.png)

### 1.2 Model Zoo:

**LLaVA-v1.5模型权重:**

| Version   | Size | Schedule   | Checkpoint                                                   | VQAv2 | GQA  | VizWiz | SQA  | T-VQA | POPE | MME    | MM-Bench | MM-Bench-CN | SEED | LLaVA-Bench-Wild | MM-Vet |
| --------- | ---- | ---------- | ------------------------------------------------------------ | ----- | ---- | ------ | ---- | ----- | ---- | ------ | -------- | ----------- | ---- | ---------------- | ------ |
| LLaVA-1.5 | 7B   | full_ft-1e | [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) | 78.5  | 62.0 | 50.0   | 66.8 | 58.2  | 85.9 | 1510.7 | 64.3     | 58.3        | 58.6 | 65.4             | 31.1   |
| LLaVA-1.5 | 13B  | full_ft-1e | [liuhaotian/llava-v1.5-13b](https://huggingface.co/liuhaotian/llava-v1.5-13b) | 80.0  | 63.3 | 53.6   | 71.6 | 61.3  | 85.9 | 1531.3 | 67.7     | 63.6        | 61.6 | 72.5             | 36.1   |
| LLaVA-1.5 | 7B   | lora-1e    | [liuhaotian/llava-v1.5-7b-lora](https://huggingface.co/liuhaotian/llava-v1.5-7b-lora) | 79.1  | 63.0 | 47.8   | 68.4 | 58.2  | 86.4 | 1476.9 | 66.1     | 58.9        | 60.1 | 67.9             | 30.2   |
| LLaVA-1.5 | 13B  | lora-1e    | [liuhaotian/llava-v1.5-13b-lora](https://huggingface.co/liuhaotian/llava-v1.5-13b-lora) | 80.0  | 63.3 | 58.9   | 71.2 | 60.2  | 86.7 | 1541.7 | 68.5     | 61.5        | 61.3 | 69.5             | 38.3   |

![image-20231027175117102](C:\Users\KevinGeorge\AppData\Roaming\Typora\typora-user-images\image-20231027175117102.png)

### 1.3 Demo演示：

下图是LLaVA对比GPT-V，BLIP2及OpenFlamingo等多模态视觉大模型，可以看出LLaVA对图像的理解比较深入

![img](https://llava-vl.github.io/images/cmp_ironing.png)

下面使用自己的图片来测试性能：对图像整体的把控是比较到位的，可以准确且详细地描述图像中的场景，但是当我聚焦到询问车牌是多少时，回答有一些瑕疵。

![image-20231027182040727](C:\Users\KevinGeorge\AppData\Roaming\Typora\typora-user-images\image-20231027182040727.png)

**PS: 支持中文问答但是经过测试，中文语境下的能力远远弱于英文语境**

## 2. MiniGPT-v2

**GitHub repo link:**

https://github.com/Vision-CAIR/MiniGPT-4

**Project Page:**

https://minigpt-v2.github.io/

**Demo:**

https://minigpt-v2.github.io/#

### 2.1 About:

Open-sourced codes for MiniGPT-4 and MiniGPT-v2. **MiniGPT-v2: Large Language Model as a Unified Interface for Vision-Language Multi-task Learning**

![image-20231027182555791](C:\Users\KevinGeorge\AppData\Roaming\Typora\typora-user-images\image-20231027182555791.png)

### 2.2 Model Zoo:

**The pretrained LLM weights**

**MiniGPT-v2** is based on Llama2 Chat 7B. For **MiniGPT-4**, we have both Vicuna V0 and Llama 2 version. Download the corresponding LLM weights from the following huggingface space via clone the repository using git-lfs.

| Llama 2 Chat 7B                                              | Vicuna V0 13B                                                | Vicuna V0 7B                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Download](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main) | [Downlad](https://huggingface.co/Vision-CAIR/vicuna/tree/main) | [Download](https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main) |

Then, set the variable *llama_model* in the model config file to the LLM weight path.

- For MiniGPT-v2, set the LLM path [here](https://github.com/Vision-CAIR/MiniGPT-4/blob/main/minigpt4/configs/models/minigpt_v2.yaml#L15) at Line 14.
- For MiniGPT-4 (Llama2), set the LLM path [here](https://github.com/Vision-CAIR/MiniGPT-4/blob/main/minigpt4/configs/models/minigpt4_llama2.yaml#L15) at Line 15.
- For MiniGPT-4 (Vicuna), set the LLM path [here](https://github.com/Vision-CAIR/MiniGPT-4/blob/main/minigpt4/configs/models/minigpt4_vicuna0.yaml#L18) at Line 18

**The pretrained model checkpoints**

Download the pretrained model checkpoints

| MiniGPT-v2 (after stage-2)                                   | MiniGPT-v2 (after stage-3)                                   | MiniGPT-v2 (online developing demo)                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Download](https://drive.google.com/file/d/1Vi_E7ZtZXRAQcyz4f8E6LtLh2UXABCmu/view?usp=sharing) | [Download](https://drive.google.com/file/d/1jAbxUiyl04SFJMN4sF1vvUU69Etuz4qa/view?usp=sharing) | [Download](https://drive.google.com/file/d/1aVbfW7nkCSYx99_vCRyP1sOlQiWVSnAl/view?usp=sharing) |

For **MiniGPT-v2**, set the path to the pretrained checkpoint in the evaluation config file in [eval_configs/minigptv2_eval.yaml](https://github.com/Vision-CAIR/MiniGPT-4/blob/main/eval_configs/minigptv2_eval.yaml#L10) at Line 8.

| MiniGPT-4 (Vicuna 13B)                                       | MiniGPT-4 (Vicuna 7B)                                        | MiniGPT-4 (LLaMA-2 Chat 7B)                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Download](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link) | [Download](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing) | [Download](https://drive.google.com/file/d/11nAPjEok8eAGGEG1N2vXo3kBLCg0WgUk/view?usp=sharing) |

For **MiniGPT-4**, set the path to the pretrained checkpoint in the evaluation config file in [eval_configs/minigpt4_eval.yaml](https://github.com/Vision-CAIR/MiniGPT-4/blob/main/eval_configs/minigpt4_eval.yaml#L10) at Line 8 for Vicuna version or [eval_configs/minigpt4_llama2_eval.yaml](https://github.com/Vision-CAIR/MiniGPT-4/blob/main/eval_configs/minigpt4_llama2_eval.yaml#L10) for LLama2 version.

### 2.3 Demo演示：

`MiniGPT-v2 Demo`一共支持`6`个`Task`：`No Tag, Grounding, Refer, Detection, Identify, VQA`.

For Abilities Involving Visual Grounding:

- Grounding: CLICK **Send** to generate a grounded image description.

- Refer: Input a referring object and CLICK **Send**.

- Detection: Write a caption or phrase, and CLICK **Send**.

- Identify: Draw the bounding box on the uploaded image window and CLICK **Send** to generate the bounding box. (CLICK "clear" button before re-drawing next time).

- VQA: Input a visual question and CLICK **Send**.

- No Tag: Input whatever you want and CLICK **Send** without any tagging

You can also simply chat in free form!



Demo中也给出了一些子任务的例子，可以直接点击来进行测试：

![image-20231027183207289](C:\Users\KevinGeorge\AppData\Roaming\Typora\typora-user-images\image-20231027183207289.png)

**Detection功能演示：**

使用Detection功能来对车牌进行目标检测，可以看到很好地完成了这个任务。

![image-20231027183438316](C:\Users\KevinGeorge\AppData\Roaming\Typora\typora-user-images\image-20231027183438316.png)

**Grounding功能展示：**

该功能是对图像的场景做一个全面的分析

![image-20231027184204112](C:\Users\KevinGeorge\AppData\Roaming\Typora\typora-user-images\image-20231027184204112.png)

**Identify功能展示：**

在原图上手动圈出一个物体，然后执行Identify功能，可以识别出圈出的物体是car

![image-20231027184634515](C:\Users\KevinGeorge\AppData\Roaming\Typora\typora-user-images\image-20231027184634515.png)

**VQA功能展示：**

很明显，VQA的功能不如LLaVA，但是比LLaVA多了目标检测等功能

![image-20231027185059064](C:\Users\KevinGeorge\AppData\Roaming\Typora\typora-user-images\image-20231027185059064.png)

## 3. fuyu-8b -- Transformer一作

**Huggingface repo link:**

https://huggingface.co/adept/fuyu-8b

### 3.1 About：

从官方页面的介绍来看，该模型并不具备精细化视觉能力，它的切入点在于速度快，易于在消费级产品上使用，个人觉得应该不符合我们项目的需求

![image-20231027185559604](C:\Users\KevinGeorge\AppData\Roaming\Typora\typora-user-images\image-20231027185559604.png)

### 3.2 Benchmarks:

根据给出的在benchmark上的数据，Fuyu-8B在AI2D上达到了SOTA，但是我们需要的是VQA能力

| val Task      | Fuyu-8B | Fuyu-Medium | LLaVA 1.5 (13.5B) | QWEN-VL (10B) | PALI-X (55B) | PALM-e-12B | PALM-e-562B |
| ------------- | ------- | ----------- | ----------------- | ------------- | ------------ | ---------- | ----------- |
| VQAv2         | 74.2    | 77.4        | 80                | 79.5          | 86.1         | 76.2       | 80.0        |
| OKVQA         | 60.6    | 63.1        | n/a               | 58.6          | 66.1         | 55.5       | 66.1        |
| COCO Captions | 141     | 138         | n/a               | n/a           | 149          | 135        | 138         |
| AI2D          | 64.5    | 73.7        | n/a               | 62.3          | 81.2         | n/a        | n/a         |

### 3.3 How to use:

**load the model and perform inference as follows:**

```python
from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image

# load model and processor
model_id = "adept/fuyu-8b"
processor = FuyuProcessor.from_pretrained(model_id)
model = FuyuForCausalLM.from_pretrained(model_id, device_map="cuda:0")

# prepare inputs for the model
text_prompt = "Generate a coco-style caption.\n"
image_path = "bus.png"  # https://huggingface.co/adept-hf-collab/fuyu-8b/blob/main/bus.png
image = Image.open(image_path)

inputs = processor(text=text_prompt, images=image, return_tensors="pt")
for k, v in inputs.items():
    inputs[k] = v.to("cuda:0")

# autoregressively generate text
generation_output = model.generate(**inputs, max_new_tokens=7)
generation_text = processor.batch_decode(generation_output[:, -7:], skip_special_tokens=True)
assert generation_text == ['A bus parked on the side of a road.']
```

**Fuyu can also perform some question answering on natural images and charts/diagrams (thought fine-tuning may be required for good performance):**

```python
text_prompt = "What color is the bus?\n"
image_path = "bus.png"  # https://huggingface.co/adept-hf-collab/fuyu-8b/blob/main/bus.png
image_pil = Image.open(image_path)

model_inputs = processor(text=text_prompt, images=[image_pil], device="cuda:0")
for k, v in model_inputs.items():
    model_inputs[k] = v.to("cuda:0")

generation_output = model.generate(**model_inputs, max_new_tokens=6)
generation_text = processor.batch_decode(generation_output[:, -6:], skip_special_tokens=True)
assert generation_text == ["The bus is blue.\n"]


text_prompt = "What is the highest life expectancy at birth of male?\n"
image_path = "chart.png"  # https://huggingface.co/adept-hf-collab/fuyu-8b/blob/main/chart.png
image_pil = Image.open(image_path)

model_inputs = processor(text=text_prompt, images=[image_pil], device="cuda:0")
for k, v in model_inputs.items():
    model_inputs[k] = v.to("cuda:0")

generation_output = model.generate(**model_inputs, max_new_tokens=16)
generation_text = processor.batch_decode(generation_output[:, -16:], skip_special_tokens=True)
assert generation_text == ["The life expectancy at birth of males in 2018 is 80.7.\n"]
```



## 4. PALI-3

谷歌发布的5B参数视觉语言模型PaLI-3，1/10体量就达到SOTA，更小更快且更强，但是**不开源**

![image-20231027190859678](C:\Users\KevinGeorge\AppData\Roaming\Typora\typora-user-images\image-20231027190859678.png)

![image-20231027190923956](C:\Users\KevinGeorge\AppData\Roaming\Typora\typora-user-images\image-20231027190923956.png)



## 5. 清华智谱CogVLM

**GitHub repo link:**

https://github.com/THUDM/CogVLM

**Project Page:**

https://chatglm.cn/

**Demo:**

http://36.103.203.44:7861/

### 5.1 About:

- CogVLM 是一个强大的开源视觉语言模型（VLM）。CogVLM-17B 拥有 100 亿视觉参数和 70 亿语言参数。
- CogVLM-17B 在 10 个经典跨模态基准测试上取得了 SOTA 性能，包括 NoCaps、Flicker30k captioning、RefCOCO、RefCOCO+、RefCOCOg、Visual7W、GQA、ScienceQA、VizWiz VQA 和 TDIUC，而在 VQAv2、OKVQA、TextVQA、COCO captioning 等方面则排名第二，超越或与 PaLI-X 55B 持平。您可以通过线上 [demo](http://36.103.203.44:7861/) 体验 CogVLM 多模态对话。

![img](https://github.com/THUDM/CogVLM/raw/main/assets/metrics-min.png)

### 5.2 对比：

- CogVLM 能够准确地描述图像，**几乎不会出现幻觉**。

 **与LLAVA-1.5 和 MiniGPT-4 的比较:**

https://raw.githubusercontent.com/THUDM/CogVLM/main/assets/llava-comparison-min.png

- CogVLM 能理解和回答各种类型的问题，并有一个**视觉定位**版本。

![img](https://github.com/THUDM/CogVLM/raw/main/assets/pear_grounding.png)

- CogVLM 有时比 GPT-4V(ision) 提取到更多的细节信息。

![img](https://github.com/THUDM/CogVLM/raw/main/assets/compare-min.png)

### 5.3 Demo演示：

这是最接近真实车牌的回答，只错了一个字母

![image-20231027192713229](C:\Users\KevinGeorge\AppData\Roaming\Typora\typora-user-images\image-20231027192713229.png)

## 6. GPT4-Vision

**很强，但CloseAI**

## 7. PALM-E

**很强，但不开源**

## 8. SoM-GPT4V

**GitHub repo link:**

https://github.com/microsoft/SoM

**Project Page:**

https://som-gpt4v.github.io/

### 8.1 About:

Set-of-Mark Prompting for LMMs. [Set-of-Mark Prompting or GPT-4V - Visual Prompting for Vision!](https://github.com/microsoft/SoM#set-of-mark-prompting-or-gpt-4v---visual-prompting-for-vision)

![image-20231027193206520](C:\Users\KevinGeorge\AppData\Roaming\Typora\typora-user-images\image-20231027193206520.png)

### 8.2 Quick Start:

### [🚀 Quick Start](https://github.com/microsoft/SoM#rocket-quick-start)

- Install segmentation packages

```bash
# install SEEM
pip install git+https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once.git@package
# install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git
# install Semantic-SAM
pip install git+https://github.com/UX-Decoder/Semantic-SAM.git@package
```

- Download the pretrained models

```bash
sh download_ckpt.sh
```

- Run the demo

```python
python demo_som.py
```

And you will see this interface:

![image-20231027193257134](C:\Users\KevinGeorge\AppData\Roaming\Typora\typora-user-images\image-20231027193257134.png)

### 8.2 Demo演示：

该项目的主要功能不是从模型推理端做视觉算法的改进，而是从数据端做数据增强操作，比如给出一张图，SoM就可以输出一个带有很多特征标记的特征图，而这种图对于视觉语言模型来进行VQA和Image Caption是非常有帮助的，可以帮助视觉模型充分释放它的能力：

**Example 1：**

![image-20231027193555709](C:\Users\KevinGeorge\AppData\Roaming\Typora\typora-user-images\image-20231027193555709.png)

**Example 2：**

![image-20231027193646370](C:\Users\KevinGeorge\AppData\Roaming\Typora\typora-user-images\image-20231027193646370.png)

**自己上传图像进行图像增强标记：**

效果看起来确实很好，之后我们进行演示的图像可以先使用SoM进行图像信息标记增强，之后再输入到视觉模型中，这样可以大大提高模型效果。

![image-20231027193943396](C:\Users\KevinGeorge\AppData\Roaming\Typora\typora-user-images\image-20231027193943396.png)



