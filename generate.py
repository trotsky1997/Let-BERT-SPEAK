import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import random
import numpy as np
# 模型名称
model_id = "google-bert/bert-base-multilingual-cased"

# 加载 tokenizer 和 模型
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# =============================
# 🔧 生成函数定义
# =============================
def sample_from_logits(logits, temperature=1.0, top_p=0.9, top_k=50, banned_token_ids=None):
    """从 logits 中执行 top-k + top-p + temperature 采样"""
    # 屏蔽禁止词
    if banned_token_ids is not None and len(banned_token_ids) > 0:
        logits = logits.clone()
        logits[banned_token_ids] = float('-inf')
    
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    # top-k
    if top_k > 0:
        top_k = min(top_k, probs.size(-1))
        topk_probs, topk_indices = torch.topk(probs, top_k)
    else:
        topk_probs, topk_indices = probs, torch.arange(probs.size(-1), device=probs.device)

    # top-p (nucleus)
    sorted_probs, sorted_indices = torch.sort(topk_probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumulative_probs > top_p
    if torch.any(cutoff):
        last_index = torch.where(cutoff)[0][0]
        sorted_probs = sorted_probs[:last_index]
        sorted_indices = sorted_indices[:last_index]

    if sorted_probs.numel() == 0:
        return topk_indices[0].item()

    sorted_probs = sorted_probs / sorted_probs.sum()
    sampled_token = torch.multinomial(sorted_probs, 1)
    return topk_indices[sorted_indices[sampled_token]].item()

def blockwise_generate_with_remask(
        model, tokenizer, input_ids, block_size=4, max_steps=5,
        temperature=1.0, top_p=0.9, top_k=50,
        remaskable_mask=None, banned_words=None
):
    device = next(model.parameters()).device
    upbound = 0.95
    lowbound = 0.95
    mask_token_id = tokenizer.mask_token_id
    mask_ids = torch.tensor([mask_token_id] * block_size, device=device, dtype=torch.long).unsqueeze(0)
    input_ids_tensor = torch.cat([input_ids, mask_ids], dim=1)
    inputs = {"input_ids": input_ids_tensor.long()}
    
    # 处理禁止词，将其转换为 token IDs
    banned_token_ids = []
    if banned_words:
        for word in banned_words:
            word_tokens = tokenizer.encode(word, add_special_tokens=False)
            banned_token_ids.extend(word_tokens)
        banned_token_ids = list(set(banned_token_ids))  # 去重
        print(f"Banned token IDs: {banned_token_ids} (for words: {banned_words})")


    # 如果是 list，转换为 tensor
    if isinstance(remaskable_mask, list):
        remaskable_mask = torch.tensor(remaskable_mask, dtype=torch.bool, device=device)
    
    # 确保 mask 长度匹配
    if remaskable_mask.size(0) != inputs["input_ids"].size(1):
        # 如果长度不匹配，扩展 mask（新添加的 token 默认可以 remask）
        extended_mask = torch.ones(inputs["input_ids"].size(1), dtype=torch.bool, device=device)
        extended_mask[:remaskable_mask.size(0)] = remaskable_mask
        remaskable_mask = extended_mask
    
    remaskable_positions = torch.tensor([
        i for i in range(inputs["input_ids"].size(1)) if remaskable_mask[i] or inputs["input_ids"][0,i].item() == tokenizer.mask_token_id
    ], device=device)

    print(f"Remaskable positions: {remaskable_positions}")
    for step in range(max_steps):
        remask_ratio = np.linspace(0, 0.9, max_steps)[-step]
        print(f"Remask ratio: {remask_ratio}")
        print(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False))
        masked_positions = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]

        for pos in masked_positions:
            sampled_id = sample_from_logits(
                logits[pos],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                banned_token_ids=banned_token_ids if banned_token_ids else None
            )
            if torch.exp(logits[pos][sampled_id]).item() > upbound:
                inputs["input_ids"][0][pos] = torch.tensor(sampled_id, dtype=torch.long, device=device)

        text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)


        sampled_pos = random.sample(remaskable_positions.tolist(), int(remask_ratio * len(remaskable_positions)))
        for pos in sampled_pos:
            inputs["input_ids"][0][pos] = torch.tensor(tokenizer.mask_token_id, dtype=torch.long, device=device)

        for pos in remaskable_positions:
            current_token_id = inputs["input_ids"][0][pos].item()
            if current_token_id != tokenizer.mask_token_id:
                # Get the confidence of the current token
                token_prob = torch.exp(logits[pos][current_token_id]).item()
                if token_prob < lowbound:
                    inputs["input_ids"][0][pos] = torch.tensor(tokenizer.mask_token_id, dtype=torch.long, device=device)


        print(f"[Step {step+1}] {text}")
    print(f"Final IDs: {inputs['input_ids'][0]}")
    return inputs["input_ids"], text


def generate_arbitrary_length(
    model, tokenizer, prompt, target_length=100, block_size=4, max_steps=5,
    temperature=1.0, top_p=0.9, top_k=50, additional_steps=0, banned_words=None
):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    input_ids = input_ids[:, :-1]
    prompt_length = input_ids.size(1)
    for step in range(target_length//block_size + additional_steps):
        current_tokens = input_ids.size(1)
        input_ids, generated_text = blockwise_generate_with_remask(
            model,
            tokenizer,
            input_ids,
            block_size=block_size if step < target_length//block_size else 0,     # 每次生成 4 个 token
            max_steps=max_steps,      # 最多循环 5 次
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            remaskable_mask=[False] * (prompt_length + max(0,step - 4) * block_size) + [True] * max(0, current_tokens - prompt_length - max(0,step - 4) * block_size),
            banned_words=banned_words
        )
        if input_ids[0, -1] == tokenizer.sep_token_id:
            input_ids[0, -1] = tokenizer.mask_token_id
    return input_ids, generated_text



# # =============================
# # 🧠 示例使用
# # =============================
# prompt = "USER#: \nHello, how are you? \n\n\nASSISTANT#: \n"
# input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
# input_ids = input_ids[:, :-1]
# prompt_length = input_ids.size(1)
# current_tokens = input_ids.size(1)
# generated = blockwise_generate_with_remask(
#     model,
#     tokenizer,
#     input_ids,
#     block_size=16,     # 每次生成 4 个 token
#     max_steps=16,      # 最多循环 5 次
#     temperature=0.8,
#     top_p=0.9,
#     top_k=20,
#     remaskable_mask=[False] * prompt_length + [True] * (current_tokens - prompt_length)
# )

# print("\n=== Final Generated ===")
# print(generated)


# =============================
# 🧠 示例使用
# =============================
prompt = "Albert Einstein was "
input_ids, generated_text = generate_arbitrary_length(
    model,
    tokenizer,
    prompt,
    target_length=128,
    block_size=4,
    max_steps=32,
    temperature=0.9,
    top_p=0.8,
    top_k=20,
    additional_steps=0,
    banned_words=["bot", "user", "[UNK]"]  # 禁止生成这些词
)
print("\n=== Final Generated ===")
print(generated_text)




