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








def process_and_filter_jsonl(input_jsonl_file_path, output_jsonl_file_path_similarity):
    heap = []
    all_entries = []

    # 逐行读取 JSONL 文件并计算相似度
    with open(input_jsonl_file_path, 'r') as file:
        for line in tqdm(file, desc="Processing JSONL file"):
            entry = json.loads(line)

            # image_path = "/mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/input/pretrain_stage_2/" + entry.get("images", [])[0]
            
            image_path = "/mnt/lustre/caipengxiang/project/better_synth/sd/" + entry.get("images", [])[0]
            
            
            # image_path = entry.get("images", [])[0]
            
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


            # 保存所有条目以便后续筛选
            all_entries.append({
                "id": entry["id"],
                "image_path": image_path,
                "text": cleaned_text,
                "clip_similarity": clip_similarity
            })

    
    # 使用 heapq 选择相似度最高的 top_n 条数据
    # top_entries = heapq.nlargest(top_n, all_entries, key=lambda x: x["clip_similarity"])

    
    
    # 将结果写入新的 JSONL 文件
    # with open(output_jsonl_file_path, 'w') as file:
    #     for entry in top_entries:
    #         # 恢复到原始的 JSON 结构格式
    #         output_entry = {
    #             "id": entry["id"],
    #             "text": f"<__dj__image>\n{entry['text']} <|__dj__eoc|>",
    #             "images": [entry["image_path"].split("pretrain_stage_1/")[1]]
    #         }
    #         file.write(json.dumps(output_entry) + '\n')


    # 将结果写入新的 JSONL 文件
    with open(output_jsonl_file_path_similarity, 'w') as file:
        for entry in all_entries:
            # 恢复到原始的 JSON 结构格式
            output_entry = {
                "id": entry["id"],
                "text": f"<__dj__image>\n{entry['text']} <|__dj__eoc|>",
                "images": [entry["image_path"].split("better_synth/sd/")[1]],
                "clip_similarity": entry["clip_similarity"]
            }
            file.write(json.dumps(output_entry) + '\n')



# 处理 JSONL 文件并保存结果
input_jsonl_file_path = "/mnt/lustre/caipengxiang/project/better_synth/sd/clip-5k+recaption-5k_simi-sort_sd_10k.jsonl"  # 替换为你的文件路径
output_jsonl_file_path_similarity = "/mnt/lustre/caipengxiang/project/better_synth/sd/clip-5k+recaption-5k_simi-sort_sd_10k-simi.jsonl"
process_and_filter_jsonl(input_jsonl_file_path, output_jsonl_file_path_similarity)
