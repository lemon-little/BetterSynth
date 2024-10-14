# Better Synth方案 ——打赢baseline就算成功

1. 赛事链接：https://tianchi.aliyun.com/competition/entrance/532251

2. 目标：本次比赛关注于多模态大模型在图片理解任务上的能力，**核心任务是在给定的种子数据集的基础上，通过高效的数据合成方法与模型生成出更优的数据，并在给定计算量的约束下，实现对图像理解多模态大模型的高效训练**，再一次探索“数据-模型”协同研发的优势。本次比赛基于 [Mini-Gemini](https://github.com/dvlab-research/MGM) 模型进行训练，**只关注于预训练（模态间对齐）阶段的数据合成与清洗**，指令微调阶段为固定数据集。为了选手更高效地迭代数据合成方案，本次比赛选用 MGM-2B 规模的模型作为比赛模型。

3. 评估方法：MMBench、TextVQA的提升比值

## 评估指标

1. MMBench：https://arxiv.org/pdf/2307.06281v4

   ![](<images/截屏2024-07-31 11.25.48.png>)

   MMBench contains over **3000 multiple-choice questions covering 20 different ability dimensions**, such as object localization and social reasoning, for evaluating vision-language models. Each ability dimension encompasses over 125 questions, with the quantity of questions per ability maintained at a roughly equal level. &#x20;

   * 理解：

     **覆盖广泛维度**:

     * 评估视觉语言模型在20个不同能力维度上的表现，例如物体定位、社交推理等。这要求数据具有多样性

     **题目数量充足**:

     * 每个能力维度下包含超过125个选择题，总计超过3000个问题，确保各维度的评估结果具有代表性。

2. TextVQA：https://paperswithcode.com/dataset/textvqa

   TextVQA is a dataset to benchmark visual reasoning based on text in images. TextVQA requires models to read and reason about text in images to answer questions about them. Specifically, **models need to incorporate a new modality of text present in the images and reason over it to answer TextVQA questions.**

   **https://textvqa.org/dataset/：Note:** Some of the images in OpenImages are rotated, please make sure to check the **Rotation** field in the Image IDs files for [train](https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv) and [test](https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv).

   * 理解：

     **文本推理能力**:

     * 评估模型读取并理解图像中文本信息的能力，要求模型能够正确地从图像中提取并利用文本信息来回答问题。

     **多模态理解**:

     * 需要模型同时处理图像信息和文本信息，测试模型在整合和推理来自不同模态的信息时的表现。

     **图像方向处理**:

     * 特别注意部分图像在数据集（OpenImages）中的旋转问题，确保模型在训练和测试过程中正确处理图像方向。

## 工具 

data-juicer

## 方案实施过程

1. **Baseline模型性能评估**

   | 详细方案     | score  | MMBench | TextVQA |
   | -------- | ------ | ------- | ------- |
   | 随机采样200K | 1.0119 | 0.9543  | 1.0696  |

2. **基于图相似度采样效果探索**：在查看数据集后，我们认为图文是否匹配对训练效果会产生很大的影响，因而使用data-juicer能够判断图文是否匹配的算子进行实验

   | 算子名                             | 场景         | 语言 | 描述                                 |
   | ------------------------------- | ---------- | -- | ---------------------------------- |
   | image\_text\_matching\_filter   | Multimodal | -  | 保留图像-文本的分类匹配分(基于BLIP模型)在指定范围内的样本   |
   | image\_text\_similarity\_filter | Multimodal | -  | 保留图像-文本的特征余弦相似度(基于CLIP模型)在指定范围内的样本 |

   | 详细方案                                                                  | score  | MMBench | TextVQA |
   | --------------------------------------------------------------------- | ------ | ------- | ------- |
   | 使用clip筛选相似度最高的200K                                                    | 1.9120 | 2.7190  | 1.1051  |
   | 使用blip筛选ITM最高的200K                                                    | 1.9089 | 2.7386  | 1.0791  |
   | 使用clip排序后取前300K，再使用blip排序筛选200K，和实验三的区别在于数据的顺序会不一样，最后的得分阈值的数据也会有一点不一样 | 1.8263 | 2.5687  | 1.0840  |
   | 使用clip和blip的归一化得分进行加权，筛选200K后随机打乱                                     | 1.8694 | 2.6602  | 1.0786  |

   * 计算图文相似度代码

     * clip

       ```python
             
       import json
       import torch
       from PIL import Image
       from tqdm import tqdm
       import open_clip
       import heapq


       device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


       # 初始化模型和预处理函数
       model, preprocess = open_clip.create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384')
       model.to(device)
       model.eval()
       tokenizer = open_clip.get_tokenizer('ViT-H-14')
    
       def preprocess_image(image_path):
           image = Image.open(image_path)
           return preprocess(image).unsqueeze(0).to(device)
    
       def compute_similarity(image_tensor, text_token):
           text_token = text_token.to(device)
           with torch.no_grad(), torch.cuda.amp.autocast():
               image_features = model.encode_image(image_tensor)
               text_features = model.encode_text(text_token)
               image_features /= image_features.norm(dim=-1, keepdim=True)
               text_features /= text_features.norm(dim=-1, keepdim=True)
               similarity = image_features @ text_features.T
           return similarity.item()
    
       def process_and_filter_jsonl(input_jsonl_file_path, output_jsonl_file_path, output_with_similarity_jsonl_file_path, top_n=318555):
           heap = []
           all_entries = []
    
           # 逐行读取 JSONL 文件并计算相似度
           with open(input_jsonl_file_path, 'r') as file:
               for line in tqdm(file, desc="Processing JSONL file"):
                   entry = json.loads(line)
    
                   # image_path = "/mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/input/pretrain_stage_1/" + entry.get("images", [])[0]
    
                   image_path = entry.get("images", [])[0]
    
                   text = entry.get("text", "")
    
                   # if not image_paths:
                   #     continue
    
                   # 提取文本中的有效部分
                   start = text.find("<__dj__image>") + len("<__dj__image>")
                   end = text.find("<|__dj__eoc|>")
                   if start == -1 or end == -1:
                       continue
                   cleaned_text = text[start:end].strip()
    
                   # 处理每对图像和文本
                   # for image_path in image_paths:
                   image_tensor = preprocess_image(image_path)
                   text_token = tokenizer(cleaned_text)
    
                   similarity = compute_similarity(image_tensor, text_token)
    
                   # 保存所有条目以便后续筛选
                   all_entries.append({
                       "id": entry["id"],
                       "image_path": image_path,
                       "text": cleaned_text,
                       "similarity": similarity
                   })
    
           # 使用 heapq 选择相似度最高的 top_n 条数据
           top_entries = heapq.nlargest(top_n, all_entries, key=lambda x: x["similarity"])
    
           # 将结果写入新的 JSONL 文件
           with open(output_jsonl_file_path, 'w') as file:
               for entry in top_entries:
                   # 恢复到原始的 JSON 结构格式
                   output_entry = {
                       "id": entry["id"],
                       "text": f"<__dj__image>\n{entry['text']} <|__dj__eoc|>",
                       "images": [entry["image_path"].split("output-08-11/en/")[1]]
                   }
                   file.write(json.dumps(output_entry) + '\n')
    
           # 将结果写入新的 JSONL 文件
           with open(output_with_similarity_jsonl_file_path, 'w') as file:
               for entry in top_entries:
                   # 恢复到原始的 JSON 结构格式
                   output_entry = {
                       "id": entry["id"],
                       "text": f"<__dj__image>\n{entry['text']} <|__dj__eoc|>",
                       "images": [entry["image_path"].split("output-08-11/en/")[1]],
                       "similarity": entry["similarity"]
                   }
                   file.write(json.dumps(output_entry) + '\n')
    
       # 处理 JSONL 文件并保存结果
       input_jsonl_file_path = "/mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/output-08-11/low_similarity_filter/res.jsonl"  # 替换为你的文件路径
       output_jsonl_file_path = "/mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/output-08-11/low_similarity_filter/res-clip-318555.jsonl"
       output_with_similarity_jsonl_file_path = "/mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/output-08-11/low_similarity_filter/res-clip-318555-simi.jsonl"
       process_and_filter_jsonl(input_jsonl_file_path, output_jsonl_file_path, output_with_similarity_jsonl_file_path)


​           
​       ```
​    
     * Blip
    
       ```python
             
       import json
       import torch
       from PIL import Image
       from tqdm import tqdm
       import open_clip
       import heapq
       from transformers import BlipProcessor, BlipForConditionalGeneration
       import torch
       from PIL import Image
    
       device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



       def compute_image_text_similarity(image_path, text):
           # 加载图像
           image = Image.open(image_path).convert("RGB")
           
           # 预处理图像和文本
           inputs = processor(images=image, text=text, return_tensors="pt").to(device)
           
           # 使用模型进行推理
           outputs = model.generate(**inputs, max_new_tokens=50)
           
           # 生成图像的描述
           generated_text = processor.decode(outputs[0], skip_special_tokens=True)
           
           # 打印生成的文本描述
           print(f"Generated Text: {generated_text}")
           
           # 计算图文匹配度
           similarity_score = compute_similarity_score(generated_text, text)
           
           return similarity_score
    
       def compute_similarity_score(generated_text, input_text):
           # 这里可以使用简单的相似性计算方法，例如使用文本相似度计算库
           # 在实际应用中，你可以替换为更复杂的相似性计算方法
           from sklearn.feature_extraction.text import TfidfVectorizer
           from sklearn.metrics.pairwise import cosine_similarity
           
           tfidf_vectorizer = TfidfVectorizer().fit_transform([generated_text, input_text])
           cosine_sim = cosine_similarity(tfidf_vectorizer[0:1], tfidf_vectorizer[1:2])
           
           return cosine_sim[0][0]




       # 初始化模型和预处理函数
    
       processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
       model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
       model.to(device)
       model.eval()



       def process_and_filter_jsonl(input_jsonl_file_path, output_jsonl_file_path, top_n=200000):
           heap = []
           all_entries = []
    
           # 逐行读取 JSONL 文件并计算相似度
           with open(input_jsonl_file_path, 'r') as file:
               for line in tqdm(file, desc="Processing JSONL file"):
                   entry = json.loads(line)
    
                   image_path = "/mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/input/pretrain_stage_1/" + entry.get("images", [])[0]
    
                   text = entry.get("text", "")
    
                   # if not image_paths:
                   #     continue
    
                   # 提取文本中的有效部分
                   start = text.find("<__dj__image>") + len("<__dj__image>")
                   end = text.find("<|__dj__eoc|>")
                   if start == -1 or end == -1:
                       continue
                   cleaned_text = text[start:end].strip()
    
                   # 处理每对图像和文本
                   # for image_path in image_paths:
                   similarity = compute_image_text_similarity(image_path, cleaned_text)
                   print(f'similarity: {similarity}')
    
                   # 保存所有条目以便后续筛选
                   all_entries.append({
                       "id": entry["id"],
                       "image_path": image_path,
                       "text": cleaned_text,
                       "similarity": similarity
                   })
    
           # 使用 heapq 选择相似度最高的 top_n 条数据
           top_entries = heapq.nlargest(top_n, all_entries, key=lambda x: x["similarity"])
    
           # 将结果写入新的 JSONL 文件
           with open(output_jsonl_file_path, 'w') as file:
               for entry in top_entries:
                   # 恢复到原始的 JSON 结构格式
                   output_entry = {
                       "id": entry["id"],
                       "text": f"<__dj__image>\n{entry['text']} <|__dj__eoc|>",
                       "images": [entry["image_path"].split("pretrain_stage_1/")[1]]
                   }
                   file.write(json.dumps(output_entry) + '\n')
    
       # 处理 JSONL 文件并保存结果
       input_jsonl_file_path = "/mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/input/pretrain_stage_1/mgm_pretrain_stage_1.jsonl"  # 替换为你的文件路径
       output_jsonl_file_path = "/mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/input/pretrain_stage_1/mgm_pretrain_stage_1-blip-200k.jsonl"
       process_and_filter_jsonl(input_jsonl_file_path, output_jsonl_file_path)


​           
​       ```
​    
     * Clip add blip
    
       ```python
             
       import json
       import torch
       from PIL import Image
       from tqdm import tqdm
       import open_clip
       import heapq
       from transformers import BlipProcessor, BlipForConditionalGeneration
       import torch
       from PIL import Image
       from sklearn.preprocessing import MinMaxScaler


       device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


       # 初始化clip模型和预处理函数
       clip_model, clip_preprocess = open_clip.create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384')
       clip_model.to(device)
       clip_model.eval()
       clip_tokenizer = open_clip.get_tokenizer('ViT-H-14')



       # 初始化blip模型和预处理函数
       blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
       blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
       blip_model.to(device)
       blip_model.eval()




       def preprocess_clip_image(image_path):
           image = Image.open(image_path)
           return clip_preprocess(image).unsqueeze(0).to(device)
    
       def compute_clip_similarity(image_tensor, text_token):
           text_token = text_token.to(device)
           with torch.no_grad(), torch.cuda.amp.autocast():
               image_features = clip_model.encode_image(image_tensor)
               text_features = clip_model.encode_text(text_token)
               image_features /= image_features.norm(dim=-1, keepdim=True)
               text_features /= text_features.norm(dim=-1, keepdim=True)
               similarity = image_features @ text_features.T
           return similarity.item()



       def compute_image_text_blip_similarity(image_path, text):
           # 加载图像
           image = Image.open(image_path).convert("RGB")
           
           # 预处理图像和文本
           inputs = blip_processor(images=image, text=text, return_tensors="pt").to(device)
           
           # 使用模型进行推理
           outputs = blip_model.generate(**inputs, max_new_tokens=50)
           
           # 生成图像的描述
           generated_text = blip_processor.decode(outputs[0], skip_special_tokens=True)
           
           # 打印生成的文本描述
           # print(f"Generated Text: {generated_text}")
           
           # 计算图文匹配度
           similarity_score = compute_similarity_score(generated_text, text)
           
           return similarity_score
    
       def compute_similarity_score(generated_text, input_text):
           # 这里可以使用简单的相似性计算方法，例如使用文本相似度计算库
           # 在实际应用中，你可以替换为更复杂的相似性计算方法
           from sklearn.feature_extraction.text import TfidfVectorizer
           from sklearn.metrics.pairwise import cosine_similarity
           
           tfidf_vectorizer = TfidfVectorizer().fit_transform([generated_text, input_text])
           cosine_sim = cosine_similarity(tfidf_vectorizer[0:1], tfidf_vectorizer[1:2])
           
           return cosine_sim[0][0]


       def normalize_scores(entries, key):
           values = [entry[key] for entry in entries]
           scaler = MinMaxScaler()
           normalized_values = scaler.fit_transform([[v] for v in values])
           for i, entry in enumerate(entries):
               entry[key] = normalized_values[i][0]








       def process_and_filter_jsonl(input_jsonl_file_path, output_jsonl_file_path, top_n=362644):
           heap = []
           all_entries = []
    
           # 逐行读取 JSONL 文件并计算相似度
           with open(input_jsonl_file_path, 'r') as file:
               for line in tqdm(file, desc="Processing JSONL file"):
                   entry = json.loads(line)
    
                   # image_path = "/mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/input/pretrain_stage_1/" + entry.get("images", [])[0]
                   image_path = entry.get("images", [])[0]
                   text = entry.get("text", "")
    
                   # if not image_paths:
                   #     continue
    
                   # 提取文本中的有效部分
                   start = text.find("<__dj__image>") + len("<__dj__image>")
                   end = text.find("<|__dj__eoc|>")
                   if start == -1 or end == -1:
                       continue
                   cleaned_text = text[start:end].strip()
    
                   # 处理每对图像和文本
                   # for image_path in image_paths:
                   image_tensor = preprocess_clip_image(image_path)
                   text_token = clip_tokenizer(cleaned_text)
    
                   clip_similarity = compute_clip_similarity(image_tensor, text_token)


                   # 处理每对图像和文本
                   # for image_path in image_paths:
                   blip_similarity = compute_image_text_blip_similarity(image_path, cleaned_text)


                   # 保存所有条目以便后续筛选
                   all_entries.append({
                       "id": entry["id"],
                       "image_path": image_path,
                       "text": cleaned_text,
                       "clip_similarity": clip_similarity,
                       "blip_similarity": blip_similarity
                   })


           # 归一化 similarity 分数
           normalize_scores(all_entries, "clip_similarity")
           normalize_scores(all_entries, "blip_similarity")
    
           # 计算综合相似度并排序
           for entry in all_entries:
               entry["combined_similarity"] = entry["clip_similarity"] + entry["blip_similarity"]


​           
​           # 使用 heapq 选择相似度最高的 top_n 条数据
​           top_entries = heapq.nlargest(top_n, all_entries, key=lambda x: x["combined_similarity"])


​           
​           
​           # 将结果写入新的 JSONL 文件
​           with open(output_jsonl_file_path, 'w') as file:
​               for entry in top_entries:
​                   # 恢复到原始的 JSON 结构格式
​                   output_entry = {
​                       "id": entry["id"],
​                       "text": f"<__dj__image>\n{entry['text']} <|__dj__eoc|>",
​                       "images": [entry["image_path"].split("pretrain_stage_1/")[1]]
​                   }
​                   file.write(json.dumps(output_entry) + '\n')


           # 将结果写入新的 JSONL 文件
           with open(output_jsonl_file_path_similarity, 'w') as file:
               for entry in top_entries:
                   # 恢复到原始的 JSON 结构格式
                   output_entry = {
                       "id": entry["id"],
                       "text": f"<__dj__image>\n{entry['text']} <|__dj__eoc|>",
                       "images": [entry["image_path"].split("pretrain_stage_1/")[1]],
                       "combined_similarity": entry["combined_similarity"],
                       "clip_similarity": entry["clip_similarity"],
                       "blip_similarity": entry["blip_similarity"],
                   }
                   file.write(json.dumps(output_entry) + '\n')



       # 处理 JSONL 文件并保存结果
       input_jsonl_file_path = "/mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/input/pretrain_stage_1/mgm_pretrain_stage_1_refined.jsonl"  # 替换为你的文件路径
       output_jsonl_file_path = "/mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/input/pretrain_stage_1/mgm_pretrain_stage_1_refined-clip-add-blip.jsonl"
       output_jsonl_file_path_similarity = "/mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/input/pretrain_stage_1/mgm_pretrain_stage_1_refined-clip-add-blip-similarity.jsonl"
       process_and_filter_jsonl(input_jsonl_file_path, output_jsonl_file_path)


​           
​       ```

3. **探测图文相似度对训练效果的影响：**基于图相似度采样效果探索显然可以看出，使用计算图文相似度的模型进行采样效果要好于随机采样，但仍需要探寻图文相似度上下界对训练效果的影响

   | 详细方案                | score      | MMBench | TextVQA |
   | ------------------- | ---------- | ------- | ------- |
   | Clip top1K-201K     | 1.8371     | 2.6210  | 1.0533  |
   | Clip top1.5K-201.5K | 2.0428     | 2.9608  | 1.1247  |
   | Clip top2K-202K     | **2.3521** | 3.5099  | 1.1944  |
   | Clip top3K-203K     | 1.9293     | 2.7778  | 1.0807  |
   | Clip top5K-205K     | 1.6674     | 2.3007  | 1.0340  |

4. **增加文本去重、过滤，图片去重、过滤的算子：**仅考虑图文相似度是不够的，还需要从文本质量、图片质量等方面纳对数据进行处理

   | 算子                              | 场景         | 语言     | 描述                                 |
   | ------------------------------- | ---------- | ------ | ---------------------------------- |
   | document\_deduplicator          | General    | en, zh | 通过比较 MD5 哈希值在文档级别对样本去重             |
   | document\_minhash\_deduplicator | General    | en, zh | 使用 MinHashLSH 在文档级别对样本去重           |
   | document\_simhash\_deduplicator | General    | en, zh | 使用 SimHash 在文档级别对样本去重              |
   | alphanumeric\_filter            | General    | en, zh | 保留字母数字比例在指定范围内的样本                  |
   | image\_nsfw\_filter             | Image      | -      | 保留包含NSFW分数在指定阈值之下的图像的样本            |
   | image\_shape\_filter            | Image      | -      | 保留样本中包含的图片的形状（即宽和高）在指定范围内的样本       |
   | image\_size\_filter             | Image      | -      | 保留样本中包含的图片的大小（bytes）在指定范围内的样本      |
   | image\_text\_matching\_filter   | Multimodal | -      | 保留图像-文本的分类匹配分(基于BLIP模型)在指定范围内的样本   |
   | image\_text\_similarity\_filter | Multimodal | -      | 保留图像-文本的特征余弦相似度(基于CLIP模型)在指定范围内的样本 |
   | image\_deduplicator             | Image      | -      | 使用文档之间图像的精确匹配在文档级别删除重复样本           |

   | 详细方案                                                                     | score  | MMBench | TextVQA |
   | ------------------------------------------------------------------------ | ------ | ------- | ------- |
   | 经过各种处理（mapper、deduplicator、filter等）之后，剩余300多K，从中随机采样200K                 | 1.8942 | 2.7125  | 1.0760  |
   | 经过各种处理（mapper、deduplicator、filter等）之后，剩余300多K，从中采样clip similarity最高的200K | 1.4343 | 1.8105  | 1.0582  |
   | 经过各种处理（mapper、deduplicator、filter等）之后，剩余300多K，取top20K-top220K            | 1.7901 | 2.4902  | 1.0899  |
   | 经过各种处理（mapper、deduplicator、filter等）之后，剩余300多K，取top50K-250K               | 1.4399 | 1.7909  | 1.0890  |

   * yaml配置文件

     * text\_and\_image\_preprocess.yaml

       ```yaml
       dataset_path: /mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/output-08-11/en/res.jsonl  
       export_path: /mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/output-08-11/text_and_image_preprocess/res.jsonl
       
       np: 42                                                            # number of subprocess to process your dataset
       text_keys: 'text'
       image_key: 'images' 
       image_special_token: '<__dj__image>'                                    # The special token that represents an image in the text. For LLaVA, it's "<image>". Should be aligned with the args when running conversion tools.
       eoc_special_token: '<|__dj__eoc|>'                                # The special token that represents the end of a chunk in the text. In default, it's "<|__dj__eoc|>". You can specify your own special token according to your input dataset. Should be aligned with the args when running conversion tools.


       # Filter ops
       process:
         - fix_unicode_mapper:        # original english = 399883                                   # fix unicode errors in text.
         - punctuation_normalization_mapper:         # 399883                    # normalize unicode punctuations to English punctuations.
         - alphanumeric_filter:           # 387798                        # filter text with alphabet/numeric ratio out of specific range.
             tokenization: false                                           # Whether to count the ratio of alphanumeric to the total number of tokens.
             min_ratio: 0.60                                               # the min ratio of filter range
         - character_repetition_filter:    # 379050                       # filter text with the character repetition ratio out of specific range
             rep_len: 10                                                   # repetition length for char-level n-gram
             max_ratio: 0.09373663                                         # the max ratio of filter range
         - flagged_words_filter:          # 378907                        # filter text with the flagged-word ratio larger than a specific max value
             lang: en                                                      # consider flagged words in what language
             tokenization: false                                           # whether to use model to tokenize documents
             max_ratio: 0.0    
         - perplexity_filter:         # 374810                            # filter text with perplexity score out of specific range
             lang: en                                                      # compute perplexity in what language
             max_ppl: 10000                                           # the max perplexity score to filter text
         - special_characters_filter:    # 367920                         # filter text with special-char ratio out of specific range
             min_ratio: 0.16534802           # 可以尝试调整成0                               # the min ratio of filter range
             max_ratio: 0.42023757                                          # the max ratio of filter range
         - word_repetition_filter:   #  367920                           # filter text with the word repetition ratio out of specific range
             lang: en                                                       # sample in which language
             tokenization: false                                            # whether to use model to tokenize documents
             rep_len: 10                                                    # repetition length for word-level n-gram
             max_ratio: 0.03085751                                          # the max ratio of filter range
         - image_aspect_ratio_filter:       # 360881                   # filter samples according to the aspect ratios of images (a fraction of width by height, r=w/h) in them
             min_ratio: 0.4                                              # the min aspect ratio of filter range
             max_ratio: 2.5                                                # the max aspect ratio of filter range
             any_or_all: any                                               # keep this sample when any/all images meet the filter condition
         - image_shape_filter:             # 360881                      # filter samples according to the widths and heights of images in them
             min_width: 336
             min_height: 336
             max_width: 1024                                     # The max width to keep samples.
             max_height: 1024                                    # The max height to keep samples.
             any_or_all: any                                               # keep this sample when any/all images meet the filter condition
         - image_size_filter:          # 359725                          # filter samples according to the size of images (in bytes) within them
             max_size: "124KB"                                             # the max size of filter range
             any_or_all: any                                               # keep this sample when any/all images meet the filter condition
         - image_nsfw_filter:          # 358092                           # 检测不适合工作时间看的内容
             hf_nsfw_model: 'Falconsai/nsfw_image_detection'
             score_threshold: 0.5
             mem_required: '10GB'
             any_or_all: any 
    
       ```
    
     * deduplication.yaml
    
       ```yaml
       dataset_path: /mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/output-08-11/text_and_image_preprocess/res.jsonl
       export_path: /mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/output-08-11/deduplication/res.jsonl
    
       np: 42                                                            # number of subprocess to process your dataset
       text_keys: 'text'
       image_key: 'images' 
       image_special_token: '<__dj__image>'                                    # The special token that represents an image in the text. For LLaVA, it's "<image>". Should be aligned with the args when running conversion tools.
       eoc_special_token: '<|__dj__eoc|>'                                # The special token that represents the end of a chunk in the text. In default, it's "<|__dj__eoc|>". You can specify your own special token according to your input dataset. Should be aligned with the args when running conversion tools.


       # Filter ops
       process:
         - document_minhash_deduplicator: # 358092 => 328464
             tokenization: 'space'                    # For English-like languages, we recommend to use 'space', for Chinese-like languages, we recommend to use 'character', and for multiple languages, we recommend to use 'sentencepiece'. If using 'sentencepiece', please provided the model path in the 'tokenizer_model' field.
             lowercase: true
             jaccard_threshold: 0.7
         - image_deduplicator:                 # 324803
             method: 'phash'
             consider_text: False  # 去重时是否考虑文本的hash值
       ```
    
     * low\_similarity\_filter.yaml
    
       ```yaml
             
       dataset_path: /mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/output-08-11/deduplication/res.jsonl
       export_path: /mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/output-08-11/low_similarity_filter/res.jsonl
    
       np: 4                                                            # number of subprocess to process your dataset
       text_keys: 'text'
       image_key: 'images' 
       image_special_token: '<__dj__image>'                                    # The special token that represents an image in the text. For LLaVA, it's "<image>". Should be aligned with the args when running conversion tools.
       eoc_special_token: '<|__dj__eoc|>'                                # The special token that represents the end of a chunk in the text. In default, it's "<|__dj__eoc|>". You can specify your own special token according to your input dataset. Should be aligned with the args when running conversion tools.


       process:
         - image_text_similarity_filter:     #324803 =>323994              # filter samples according to the similarity between text and images.
             hf_clip: openai/clip-vit-base-patch32                         # name of used Hugging Face clip
             min_score: 0.20315419                                         # the min similarity of filter range
             mem_required: '10GB'
             any_or_all: any 
         - image_text_matching_filter:       # 315086                            # filter samples according to the matching score between image and text.
             hf_blip: Salesforce/blip-itm-base-coco                        # name of used Hugging Face blip
             min_score: 0.44930778                                         # the min matching score of filter range
             mem_required: '10GB'
             any_or_all: any
       ```

   * remove\_chinese.py

     ```python
           
     # 配置数据处理文件
     import json
     import jsonlines as jl
     from tqdm import tqdm
     import os


     def check_path(path):
         if not os.path.exists(path):
             os.makedirs(path)
         return
    
     input_root = "/mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/input/pretrain_stage_1"
     output_root = "/mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/output-08-11"
     recipe_root = "/mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/solution/recipes"


     ## 去除中文数据
     all_entries = []
     with open(f"{input_root}/mgm_pretrain_stage_1.jsonl", "r") as file:
         for line in tqdm(file, desc="Processing JSONL file"):
             # 读取样本
             entry = json.loads(line)
             if entry["id"].startswith("allava"): continue
             all_entries.append(entry)
    
     check_path(f"{output_root}/en/")
     with open(f"{output_root}/en/res.jsonl", 'w') as file:
         for entry in all_entries:
             file.write(json.dumps(entry) + '\n')
     ```

5. **混合生成数据：**显然，在使用mapper、deduplicator、filter之后，效果都是下降的，这说明仅使用clip采样是合理的，但使用mapper、deduplicator、filter存在过滤调高质量数据的不确定因素，因而我们的采样、过滤策略到此为止，没有再进行过多的尝试，在此基础上，进行合成数据

   | 算子                        | 场景         | 语言 | 描述                                        |
   | ------------------------- | ---------- | -- | ----------------------------------------- |
   | image\_captioning\_mapper | Multimodal | -  | 生成样本，其标题是根据另一个辅助模型（例如 blip2）和原始样本中的图形生成的。 |
   | image\_diffusion\_mapper  | Multimodal | -  | 用stable diffusion生成图像，对图像进行增强             |

   | 详细方案                                                     | score                          | MMBench            | TextVQA            | clip排序代码 | recaption.yaml文件         |
   | ------------------------------------------------------------ | ------------------------------ | ------------------ | ------------------ | ------------ | -------------------------- |
   | CLIP 400K排序后，将后200K数据recaptioning后，重新和原CLIP top-200K数据组合，再进行一次CLIP 200K排序最终完整方案 | **2.4069**                     | 3.5818             | 1.2321             | code/sort.py | code/image_captioning.yaml |
   | CLIP 400K排序后，将后200K数据recaptioning后，重新和原CLIP top-200K数据组合，再进行一次clip 400k排序，分别实验clip top1k-201k，clip top2k-202k，clip5k-205k探索 | 1k  1.56632k  2.01243k  1.8113 | 2.05232.90852.5360 | 1.08031.11621.0867 | -            | -                          |

6. **聚类保证多样性：**根据以往经验，我们认为保证数据多样性，能够提高模型效果，因而在此基础上，使用聚类保证多样性

   | 详细方案                    | score  | MMBench | TextVQA |
   | ----------------------- | ------ | ------- | ------- |
   | 400k数据，400个聚类中心均匀采样200K | 2.0813 | 3.0197  | 1.1429  |

   1. 计算图片向量并保存

   2. 使用sklearn.cluster.MiniBatchKMeans实现聚类采样

      ```python
      from sklearn.cluster import MiniBatchKMeans
      from sklearn.decomposition import PCA
      from sklearn.datasets import make_blobs
      import numpy as np
      from sklearn.cluster import MiniBatchKMeans
      import json
      
      # 数据样例
      # data = [
      #     {"image_id": "1234",
      #      "__dj__stats__": {"alnum_ratio": 0.6615384615, "aspect_ratios": [0.7813953488], "char_rep_ratio": 0.0,
      #                        "flagged_words_ratio": 0.0, "image_height": [430], "image_nsfw_score": [0.0001461331],
      #                        "image_sizes": [74827], "image_text_matching_score": [0.999979496],
      #                        "image_text_similarity": [0.546875], "image_width": [336], "perplexity": 5334.7,
      #                        "special_char_ratio": 0.4, "word_rep_ratio": 0.0, "text_embedding": [1, 2, 3, 4]}, },
      #     # 其他数据条目
      # ]
      data = []
      
      n_samples, n_clusters, n_features= 400_00, 100, 2
      
      # 随机生成数据
      for i in range(n_samples):
          data.append({"image_id": f"image_{i}",
                       "__dj__stats__": {"alnum_ratio": np.random.rand(), "aspect_ratios": [np.random.rand()], "char_rep_ratio": np.random.rand(),
                                         "flagged_words_ratio": np.random.rand(), "image_height": [np.random.randint(100, 1000)], "image_nsfw_score": [np.random.rand()],
                                         "image_sizes": [np.random.randint(100, 100000)], "image_text_matching_score": [np.random.rand()],
                                         "image_text_similarity": [np.random.rand()], "image_width": [np.random.randint(100, 1000)], "perplexity": np.random.randint(1000, 10000),
                                         "special_char_ratio": np.random.rand(), "word_rep_ratio": np.random.rand(), "text_embedding": np.random.random(size=(1, n_features))[0]}, })
      
      print(data[0])
      # 提取text_embedding用于聚类
      embeddings = np.array([item["__dj__stats__"]['text_embedding'] for item in data])
      
      # 使用MiniBatchKMeans进行聚类
      minibatch_kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                                random_state=0,
                                batch_size=1024,
                                max_iter=30,
                                max_no_improvement=10, # number of mini batches that does not yield an improvement on the smoothed inertia.
                                reassignment_ratio= 0.1, # 每次更新的聚类中心数量
                                verbose=True)
      
      minibatch_kmeans.fit(embeddings)


      # 为每个数据点分配簇标签
      labels = minibatch_kmeans.labels_
    
      # 将聚类标签保存到__dj__stats__中
      for i, label in enumerate(labels):
          data[i]["__dj__stats__"]["cluster_label"] = int(label)  # 保存聚类标签
    
      # 采样个数
      sample_num = 300_00
      # 聚类后对每个类别进行排序和采样
      n = sample_num // n_clusters  # 假设每个类别采样前n个
      sampled_data = []
      remaining_data = {}


      for cluster_label in range(n_clusters):
          # 获取属于该簇的所有数据点
          cluster_items = [data[i] for i in range(len(data)) if labels[i] == cluster_label]
          print(f"Cluster {cluster_label} has {len(cluster_items)} items")
    
          # 根据image_text_similarity排序
          cluster_items.sort(key=lambda x: x["__dj__stats__"]["image_text_similarity"], reverse=True)
    
          if len(cluster_items) >= n:
              sampled_data.extend(cluster_items[:n]) 
    
              remaining_data[cluster_label]= cluster_items[n:]
          else:
              sampled_data.extend(cluster_items)
    
      remain_sample_num = sample_num - len(sampled_data)
      remain_n = (remain_sample_num + len(remaining_data) - 1) // len(remaining_data)
    
      while len(sampled_data) < sample_num:
          for cluster_label, cluster_items in remaining_data.items():
              if len(cluster_items) >= remain_n:
                  sampled_data.extend(cluster_items[:remain_n])
                  cluster_items = cluster_items[remain_n:]
                  remaining_data[cluster_label] = cluster_items
              else:
                  sampled_data.extend(cluster_items)
                  del remaining_data[cluster_label]
          remain_sample_num = sample_num - len(sampled_data)
          remain_n = (remain_sample_num + len(remaining_data) - 1) // len(remaining_data)
          print(f"Sampled {len(sampled_data)} items, remaining {remain_sample_num} items to sample")


      print(f"Sampled {len(sampled_data)} items")
      # 输出采样后的数据
      for item in sampled_data:
          print(f"Sampled Image ID: {item['image_id']}, Similarity: {item['__dj__stats__']['image_text_similarity']}")


      def save_data_as_jsonl(data, file_path):
          """
          将数据保存为jsonl格式，确保ndarray被转换为可保存的json格式。
    
          :param data: 要保存的数据列表，其中每个数据项是一个字典。
          :param file_path: 保存文件的路径。
          """
          # 将数据中的 ndarray 转换为可保存的列表格式
          for item in data:
              for key, value in item["__dj__stats__"].items():
                  if isinstance(value, np.ndarray):
                      item["__dj__stats__"][key] = value.tolist()
                  elif isinstance(value, list):
                      item["__dj__stats__"][key] = [v.tolist() if isinstance(v, np.ndarray) else v for v in value]
    
          # 将数据写入jsonl文件
          with open(file_path, 'w') as f:
              for item in data:
                  f.write(json.dumps(item) + "\n")
    
      sampled_data = sampled_data[:sample_num]  # 截取前sample_num个数据
      save_data_as_jsonl(sampled_data, "sampled_data.jsonl")
      ```

7. 未找到更好的聚类效果，初赛结束，终止记录

## 决赛方案

在初赛最佳实验结果的基础上，多选了5K预训练数据，即使用clip排序后，前200K保留，后200K recaption，再排序后选top205K，这种方案无论是在初赛还是复赛中都有很强的泛化性，比如我们的初赛模型在复赛指标上只下降了0.2分。

