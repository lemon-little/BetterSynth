import json

# Function to read the jsonl file and sort based on "clip_similarity"
def sort_jsonl_by_similarity(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Parse each line as a JSON object
    json_objects = [json.loads(line) for line in lines]

    # Sort the list of JSON objects by "clip_similarity" in descending order
    sorted_json_objects = sorted(json_objects, key=lambda x: x["clip_similarity"], reverse=True)

    # Write the sorted objects back to a new jsonl file
    with open(output_file, 'w') as file:
        for json_object in sorted_json_objects:
            file.write(json.dumps(json_object) + '\n')

    

# File paths
input_file = '/mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/input/pretrain_stage_1/clip-100k+recaption-300k.jsonl'
output_file = '/mnt/lustre/caipengxiang/project/better_synth/dj_synth_challenge/input/pretrain_stage_1/clip-100k+recaption-300k_sort.jsonl'

sort_jsonl_by_similarity(input_file, output_file)
