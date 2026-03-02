from datasets import load_dataset

ds = load_dataset("liamdugan/raid", split="train", streaming=True)



def get_n_prompts(n, dataset_id = 'liamdugan/raid'):
    collected = set()
    human_prompt_list = []
    ds = load_dataset("liamdugan/raid", split="train", streaming=True)
    human_by_source = {}
    prompt_by_source = {}
    for example in ds:
        source_id = example['source_id']
        if example['prompt']:
            prompt_by_source[source_id] = example['prompt']
        if example["model"] == 'human':
            human_by_source[source_id] = example["generation"]
        if source_id in human_by_source and source_id in prompt_by_source:
            human_text = human_by_source[source_id]
            prompt = prompt_by_source[source_id]
            if (source_id) not in collected:
                collected.add(source_id)
                human_prompt_list.append([human_text,prompt])
                n-=1

        if n == 0:
            return human_prompt_list
    
print(get_n_prompts(20))

