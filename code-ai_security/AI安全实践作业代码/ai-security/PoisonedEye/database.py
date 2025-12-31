# Database class for retrieval 

from transformers import AutoModel, AutoProcessor
from torchvision import transforms
import faiss
import torch
from PIL import Image
import numpy as np
from torchvision.utils import save_image
from utils import tensor_to_pil, get_weighted_sum, get_cluster_center, get_cluster_center_AP, \
                    pil_to_tensor, resize_image_tensor, normalize_image, denormalize_image, center_crop_tensor

class Database():
    def __init__(self, database_path, encode_model_path, device): #加载磁盘上的 FAISS 索引和指定路径的多模态编码模型与处理器，并记录设备。
        # 加载 FAISS 索引是为了快速检索。索引中包含了所有数据库项的向量表示，是实现高效相似性搜索的基础。
        self.database = faiss.read_index(database_path) # load index from disk
        self.model  = AutoModel.from_pretrained(encode_model_path).to(device) # load model
        # 加载处理器 是为了将输入数据转换为模型可以理解的格式。处理器负责对原始图像（如 PIL.Image）和文本（如字符串）进行必要的预处理，
        # 使其能够输入到编码模型中并生成向量。
        self.processor = AutoProcessor.from_pretrained(encode_model_path) # load processer 
        self.device = device
    
    def get_image_vector(self, image, encoder=None, processor=None):# 将单张图像编码为单位长度的向量，可传入自定义编码器/处理器。
        if encoder is None and processor is None:
            encoder = self.model
            processor = self.processor
        image_input = processor(images=image, return_tensors="pt", padding=True).to(self.device) # 图像预处理
        image_outputs = encoder.get_image_features(**image_input) # 图像编码
        image_outputs = image_outputs / image_outputs.norm(p=2, dim=-1, keepdim=True)# L2归一化
        return image_outputs
    
    def get_image_vector_batch(self, image, encoder=None, processor=None): # 批量图像编码，逐张无梯度地处理后堆叠返回
        if encoder is None and processor is None:
            encoder = self.model
            processor = self.processor
        res=[]
        with torch.no_grad(): # torch.no_grad是一个上下文管理器，确保在编码过程中不计算和存储梯度
            for i in range(len(image)):
                img = [image[i]]
                image_input = processor(images=img, return_tensors="pt", padding=True).to(self.device)
                image_outputs = encoder.get_image_features(**image_input)
                image_outputs = image_outputs / image_outputs.norm(p=2, dim=-1, keepdim=True)
                res.append(image_outputs.squeeze(0))
        return torch.stack(res) #将所有单独的向量堆叠成单一的张量，堆叠的好处是便于在pytorch中进行各种并行计算和批量操作
    
    # get_text_vector 方法之所以可以对单条或多条文本进行编码和归一化，是因为它利用了 Hugging Face transformers 库中 
    # AutoProcessor 的灵活性，以及它内部的 Tokenizer 机制。
    def get_text_vector(self, text, encoder=None, processor=None): #对单条或多条文本编码并归一化
        if not isinstance(text, list):
            text = [text]
        if encoder is None and processor is None:
            encoder = self.model
            processor = self.processor
        # 具体来说，Tokenizer会对列表中的每条文本进行分词，使用 padding=True，它会自动找到批次中最长的句子，然后用填充标记 ([PAD]) 填充所有其他较短的句子，
        # 使所有文本序列的长度保持一致，从而可以堆叠成一个矩阵（张量）。
        query_text = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device) #处理器内的 Tokenizer 会并行处理列表中的所有文本
        text_outputs = encoder.get_text_features(**query_text)
        text_outputs = text_outputs / text_outputs.norm(p=2, dim=-1, keepdim=True)
        return text_outputs

    def get_vector(self, text, image): #先分别编码文本和图像，再求和并归一化，返回可直接用于检索的 numpy 向量
        text_outputs = self.get_text_vector(text)
        image_outputs = self.get_image_vector(image)

        vectors = image_outputs + text_outputs
        vectors = vectors / vectors.norm(p=2, dim=-1, keepdim=True)
        vectors = vectors.detach().cpu().numpy() # convert to numpy
        return vectors
    
    def query(self, text: str, image, k=5, search_range=None): #根据传入的文本、图像或二者组合得到向量，并在 FAISS 索引中执行 KNN 或范围搜索，返回距离与索引
        if k==0:
            return [[]],[[]]
        if text is None:
            vectors = self.get_image_vector(image).detach().cpu().numpy()
        elif image is None:
            vectors = self.get_text_vector(text).detach().cpu().numpy()
        else:
            vectors = self.get_vector(text, image)
        if search_range: # 如果search_range参数为true，就执行范围搜索，所有小于search_range的邻居都会被返回。k的数量就不固定了
            lims, distances, indices = self.database.range_search(vectors, search_range) # neighbors within search_range
        else: # 返回k个距离最近的
            distances, indices = self.database.search(vectors, k) # k closest neighbors
        return distances, indices

    def add(self, text, image): #将图文向量添加到索引尾部
        vectors = self.get_vector(text, image)
        self.database.add(vectors)

    def remove_last(self): #删除索引中最新插入的条目
        length = self.database.ntotal
        index_to_remove = np.array([length-1], dtype=np.int64)
        return self.database.remove_ids(index_to_remove)

    #get_pixel_value 最重要的用途是在执行对抗性攻击（如 create_poison 方法）时，获取一个基准像素张量，以便在此基础上添加微小的扰动
    def get_pixel_value(self, image): # 使用内置处理器生成模型需要的像素张量
        image_pixel_value = self.processor(images=image, return_tensors="pt", padding=True).to(self.device)['pixel_values']
        return image_pixel_value
    
    def encode_image(self, image, processor=None): # 把图像打包成模型输入（包含张量和附加字段），作为检索模块的模型的输入
        if processor is None:
            processor = self.processor
        return processor(images=image, return_tensors="pt", padding=True).to(self.device)
    
    
    # 通过迭代梯度方法调整扰动，生成与目标图文向量高度相似的“投毒”图像，并保存为 poison.png
    # poison_image, poison_text作为攻击的起点，target_image和target_text作为攻击的目标
    # steps是迭代优化的步数
    def create_poison(self, poison_image, poison_text, target_image, target_text, steps = 1000):
        image_ref = self.encode_image(poison_image)
        image_input = self.encode_image(poison_image)
        image_pert = torch.zeros_like(image_input['pixel_values']).requires_grad_() #初始化扰动张量，全零且需要梯度

        target_text_vector = self.get_text_vector(target_text).mean(dim=0) #计算目标文本的特征向量并求平均
        poison_text_vector = self.get_text_vector(poison_text)
        target_image_vector = self.get_image_vector_batch(target_image).mean(dim=0) 
        target_vector = target_text_vector + target_image_vector
        target_vector = target_vector / target_vector.norm(p=2, dim=-1, keepdim=True) #对混合向量进行L2归一化

        best_step = 0
        best_sim = 0 
        best_pert = None # 最佳扰动

        for step in range(steps):
            image_input['pixel_values'] = image_ref['pixel_values'] + image_pert #将基准图像像素值 image_ref['pixel_values'] 与当前扰动 image_pert 相加，作为新的图像输入
            image_outputs = self.model.get_image_features(**image_input) #使用加扰动后的图像输入通过模型编码得到 “投毒”图像向量
            poison_image_vector = image_outputs / image_outputs.norm(p=2, dim=-1, keepdim=True)

            poison_vector = poison_image_vector + poison_text_vector
            poison_vector = poison_vector / poison_vector.norm(p=2, dim=-1, keepdim=True)

            sim = torch.dot(poison_vector.squeeze(0), target_vector)
            if sim > best_sim:
                best_step = step
                best_pert = image_pert
                best_sim = sim
            # if step==0 or step%100==99:
            #     print(f"Step {step} | Sim = {sim}")

            gradient = torch.autograd.grad(outputs=-sim, inputs=image_pert)[0] #目标是最大化Sim，因此我们最小化Sim

            image_pert = image_pert - 0.01 * gradient.sign() #因为我们最小化的是负相似度，所以沿着负梯度的符号方向移动（相当于沿着正相似度的正梯度方向移动），步长为 0.01
            image_pert = image_pert.clamp(max=0.0625, min=-0.0625) #将扰动的值限制在 [-0.0625, 0.0625]的范围内。这确保了扰动足够微小，使得最终的“投毒”图像与原始图像在视觉上几乎相同。

        # print(f"Best step {best_step} | Sim = {best_sim}")
        best_image_pixel = (image_ref['pixel_values'] + best_pert).squeeze(0) #使用记录的最佳扰动 best_pert 加上原始像素值，得到最终的“投毒”图像的像素张量


        img_mean = torch.tensor(self.processor.image_processor.image_mean, dtype=torch.float32)
        img_std = torch.tensor(self.processor.image_processor.image_std, dtype=torch.float32)
        pil_image = tensor_to_pil(best_image_pixel, img_mean, img_std=img_std)
        pil_image.save('poison.png')
        
        return pil_image
    
    def compute_distance(self, image1, text1, image2, text2):# 计算两组图文样本之间的综合向量 L2 距离、图像向量 L2、文本向量 L2 以及余弦相似度，返回字典结果
        image_vector1 = self.get_image_vector(image1)
        image_vector2 = self.get_image_vector(image2)
        text_vector1 = self.get_text_vector(text1)
        text_vector2 = self.get_text_vector(text2)
        vector1 = self.get_vector(text1, image1)
        vector2 = self.get_vector(text2, image2)
        # compute l2_distance
        diff = vector1[0] - vector2[0]
        distance = np.linalg.norm(diff, ord=2)
        # compute image_l2_distance
        diff = image_vector1[0] - image_vector2[0]
        image_distance = diff.norm(p=2).item()
        # compute text_l2_distance
        diff = text_vector1[0] - text_vector2[0]
        text_distance = diff.norm(p=2).item()
        # compute cos_sim  (since vectors have been normalized, |vector1|=|vector2|=1)
        sim = np.dot(vector1[0], vector2[0])
        
        res={"l2_distance": distance,
             "image_l2_distance": image_distance,
             "text_l2_distance": text_distance,
             "cos_sim": sim,}
        return res