# Create retrieval database for oven-wiki

import torch
import faiss
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from data.mbeir_dataset import (
    MBEIRMainDataset,
    MBEIRCandidatePoolDataset,
    MBEIRMainCollator,
    MBEIRCandidatePoolCollator,
    Mode,
)
from transformers import AutoProcessor, AutoModel
import argparse
import os
import json

parser = argparse.ArgumentParser(description='Create a faiss database on beir dataset\'s candidate pool')
parser.add_argument('--model_path', type=str, default='/home/model/siglip-so400m-patch14-384', help='Model path to encode texts and images')
parser.add_argument('--beir_path', type=str, default='./M-BEIR', help='Path to beir dataset')
parser.add_argument('--beir_cand_pool_path', type=str, default='cand_pool/local/mbeir_infoseek_task8_cand_pool.jsonl', help='Path to beir dataset candidate pool')
parser.add_argument('--save_path', type=str, default='clip_mbeir_infoseek_task8_cand_pool.bin', help='Path to save faiss database')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
parser.add_argument('--dim', type=int, default=768, help='Dimension of the embedding')
parser.add_argument('--max_samples', type=int, default=None, help='Max number of samples to process (for debugging/testing)')
parser.add_argument('--checkpoint_interval', type=int, default=100, help='Save checkpoint every N batches')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义checkpoint文件路径
checkpoint_path = args.save_path + '.checkpoint'
progress_path = args.save_path + '.progress.json'

processor = AutoProcessor.from_pretrained(args.model_path)
model = AutoModel.from_pretrained(args.model_path).to(device)
model = model.eval()
    
cand_pool = MBEIRCandidatePoolDataset(mbeir_data_dir=args.beir_path,
                                    cand_pool_data_path=args.beir_cand_pool_path,
                                    img_preprocess_fn=None,)

if args.max_samples is not None:
    print(f"Warning: Truncating dataset to {args.max_samples} samples!")
    cand_pool.cand_pool = cand_pool.cand_pool[:args.max_samples]

# 初始化或加载FAISS索引和进度
start_batch = 0
if os.path.exists(checkpoint_path) and os.path.exists(progress_path):
    print(f"Found checkpoint at {checkpoint_path}, resuming...")
    # 加载FAISS索引
    faiss_index = faiss.read_index(checkpoint_path)
    # 加载进度信息
    with open(progress_path, 'r') as f:
        progress = json.load(f)
        start_batch = progress['batch_idx'] + 1
        processed_samples = progress['processed_samples']
    print(f"Resuming from batch {start_batch}, {processed_samples} samples already processed.")
else:
    print("No checkpoint found, starting from scratch.")
    faiss_index = faiss.IndexFlatL2(args.dim)

def cand_pool_collator(batch):
    images = []
    texts = []
    for sample in batch:
        images.append(sample['img'])
        texts.append(sample['txt'])
    text_output = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    image_output = processor(images=images, return_tensors="pt")
    return image_output, text_output

dataloader = DataLoader(cand_pool, batch_size=args.batch_size, shuffle=False, collate_fn=cand_pool_collator, drop_last=False, num_workers=21, pin_memory=True)

total_batches = len(dataloader)
with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(dataloader, initial=start_batch, total=total_batches)):
        # 跳过已处理的batch
        if batch_idx < start_batch:
            continue
            
        image, text = batch[0].to(device), batch[1].to(device)
        text_outputs = model.get_text_features(**text)
        image_outputs = model.get_image_features(**image)
        
        # normalize model output embeddings
        image_outputs = image_outputs / image_outputs.norm(p=2, dim=-1, keepdim=True)
        text_outputs = text_outputs / text_outputs.norm(p=2, dim=-1, keepdim=True)

        # normalize the average vertors
        vectors = image_outputs + text_outputs
        vectors = vectors / vectors.norm(p=2, dim=-1, keepdim=True)
        vectors = vectors.detach().cpu().numpy() # convert to numpy

        faiss_index.add(vectors)
        
        # 定期保存checkpoint
        if (batch_idx + 1) % args.checkpoint_interval == 0:
            print(f"\nSaving checkpoint at batch {batch_idx + 1}...")
            faiss.write_index(faiss_index, checkpoint_path)
            progress = {
                'batch_idx': batch_idx,
                'processed_samples': faiss_index.ntotal,
                'total_batches': total_batches
            }
            with open(progress_path, 'w') as f:
                json.dump(progress, f)
            print(f"Checkpoint saved. Processed {faiss_index.ntotal} samples so far.")

# 保存最终结果
print(f"\nProcessing complete! Saving final index to {args.save_path}...")
faiss.write_index(faiss_index, args.save_path)

# 删除checkpoint文件
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)
    print(f"Removed checkpoint file: {checkpoint_path}")
if os.path.exists(progress_path):
    os.remove(progress_path)
    print(f"Removed progress file: {progress_path}")

print(f"Done! Final index saved to {args.save_path} with {faiss_index.ntotal} vectors.")
