
# top 200k
import json

# 输入和输出文件路径
input_file = '/mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/input/pretrain_stage_1/clip-200k+recaption-200k_clip-400k.jsonl'
output_file = '/mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/input/pretrain_stage_1/top-205k.jsonl'

# 设定要读取的行数
lines_to_read = 205000

# 打开输入文件和输出文件
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for i, line in enumerate(infile):
        if i < lines_to_read:
            outfile.write(line)
        else:
            break

print(f"Successfully saved the first {lines_to_read} lines to {output_file}")


# top 20k - top 220k
# import json

# # 输入和输出文件路径
# input_file = '/mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/output-08-13/preprocess/res_ab_clip-removes.jsonl'
# output_file = '/mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/output-08-13/preprocess/res_ab_clip-removes-top2k-202k.jsonl'

# # 设定要读取的行数
# start_line = 2000
# end_line = 202000

# # 打开输入文件和输出文件
# with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
#     for i, line in enumerate(infile):
#         if start_line <= i < end_line:
#             outfile.write(line)
#         elif i >= end_line:
#             break

# print(f"Successfully saved lines from {start_line} to {end_line} to {output_file}")


# top 50k - top 250k
# import json

# # 输入和输出文件路径
# input_file = '/mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/output-08-11/low_similarity_filter/res-clip-318555.jsonl'
# output_file = '/mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/output-08-11/low_similarity_filter/res-clip-318555-50k-250k.jsonl'

# # 设定要读取的行数
# start_line = 50000
# end_line = 250000

# # 打开输入文件和输出文件
# with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
#     for i, line in enumerate(infile):
#         if start_line <= i < end_line:
#             outfile.write(line)
#         elif i >= end_line:
#             break

# print(f"Successfully saved lines from {start_line} to {end_line} to {output_file}")