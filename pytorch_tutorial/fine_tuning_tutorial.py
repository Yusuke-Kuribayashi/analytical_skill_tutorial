import os
import numpy as np
import torch
import torchvision
import pytorch_tutorial as pt
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.pyplot as plt
import cv2
import json
import time
import psutil
from datetime import datetime
import pandas as pd



class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.nonzero(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

######################
# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 2  # 1 class (person) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


######################


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

######################
def get_transform(train):
    transforms = []
    transforms.append(pt.PILToTensor())
    transforms.append(pt.ConvertImageDtype(torch.float))
    if train:
        transforms.append(pt.RandomHorizontalFlip(0.5))
    return pt.Compose(transforms)

######################
def save_training_config(config, filename='training_config.json'):
    """学習設定を保存"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

def visualize_predictions(model, dataset, device, num_samples=5, save_dir='images/finetuning'):
    """モデルの予測結果を可視化"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    results = []
    
    for i in range(min(num_samples, len(dataset))):
        img, target = dataset[i]
        
        # 予測を実行
        with torch.no_grad():
            prediction = model([img.to(device)])
        
        # 画像をnumpy arrayに変換
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Ground truthの可視化
        gt_img = img_bgr.copy()
        if 'boxes' in target:
            for box in target['boxes']:
                x1, y1, x2, y2 = box.int().tolist()
                cv2.rectangle(gt_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(gt_img, 'GT Person', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 予測結果の可視化
        pred_img = img_bgr.copy()
        pred = prediction[0]
        if len(pred['boxes']) > 0:
            # スコアが0.5以上のものだけ表示
            high_score_idx = pred['scores'] > 0.5
            boxes = pred['boxes'][high_score_idx]
            scores = pred['scores'][high_score_idx]
            
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box.int().tolist()
                cv2.rectangle(pred_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(pred_img, f'Pred: {score:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 画像を保存
        combined_img = np.hstack([gt_img, pred_img])
        cv2.imwrite(f'{save_dir}/comparison_{i}.jpg', combined_img)
        
        # 結果を記録
        results.append({
            'image_id': i,
            'num_gt_boxes': len(target['boxes']) if 'boxes' in target else 0,
            'num_pred_boxes': len(pred['boxes'][pred['scores'] > 0.5]),
            'max_pred_score': float(pred['scores'].max()) if len(pred['scores']) > 0 else 0.0
        })
    
    return results

def plot_loss_curves(loss_history, save_path='images/finetuning/loss_curves.png'):
    """Loss曲線をプロット"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # 各lossタイプをプロット
    for loss_type in loss_history.keys():
        if loss_type != 'epoch' and len(loss_history[loss_type]) > 0:
            plt.plot(loss_history['epoch'], loss_history[loss_type], label=loss_type, marker='o')
    
    plt.xlabel('エポック')
    plt.ylabel('Loss値')
    plt.title('学習中のLoss推移')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, data_loader, device):
    """モデルの評価を実行"""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # 予測
            predictions = model(images)
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            num_batches += 1
    
    # 評価メトリクスを計算（簡単な例）
    total_pred_boxes = sum(len(pred['boxes']) for pred in all_predictions)
    total_gt_boxes = sum(len(target['boxes']) for target in all_targets)
    
    return {
        'total_predictions': total_pred_boxes,
        'total_ground_truth': total_gt_boxes,
        'num_images': len(all_predictions)
    }

######################
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
dataset = PennFudanDataset('/share/dtu_drone_tutorial/PennFudanPed', get_transform(train=True))
data_loader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=2, 
    shuffle=True, 
    num_workers=4,
    collate_fn=pt.collate_fn
)
# For Training
images,targets = next(iter(data_loader))
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]
output = model(images,targets)   # Returns losses and detections
# For inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)           # Returns predictions


######################
def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 学習設定を定義
    training_config = {
        "実験日時": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "デバイス": str(device),
        "バッチサイズ": 2,
        "学習率": 0.005,
        "使用モデル": "Mask R-CNN with ResNet-50 FPN backbone",
        "最適化アルゴリズム": "SGD",
        "モメンタム": 0.9,
        "重み減衰": 0.0005,
        "エポック数": 10,
        "学習率スケジューラ": "StepLR (step_size=3, gamma=0.1)",
        "データセット": "PennFudanPed",
        "クラス数": 2
    }
    
    # 設定を保存
    save_training_config(training_config)
    
    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = PennFudanDataset('/share/dtu_drone_tutorial/PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('/share/dtu_drone_tutorial/PennFudanPed', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=training_config["バッチサイズ"], 
        shuffle=True, 
        num_workers=4,
        collate_fn=pt.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4,
        collate_fn=pt.collate_fn
    )

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # 学習前の評価
    print("学習前の評価を実行中...")
    pre_training_eval = evaluate_model(model, data_loader_test, device)
    pre_training_vis = visualize_predictions(model, dataset_test, device, num_samples=5, save_dir='images/finetuning/pre_training')

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, 
        lr=training_config["学習率"],
        momentum=training_config["モメンタム"], 
        weight_decay=training_config["重み減衰"]
    )
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    # Loss履歴を記録するための辞書
    loss_history = {
        'epoch': []
    }
    
    # メモリとパフォーマンス記録
    performance_log = []

    # let's train it for specified epochs
    num_epochs = training_config["エポック数"]
    
    print("学習を開始します...")
    overall_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # メモリ使用量を記録
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_memory_before = torch.cuda.memory_allocated(device) / 1024**2  # MB
        
        cpu_memory_before = psutil.virtual_memory().used / 1024**2  # MB
        
        # 1エポックの学習を実行し、lossを取得
        model.train()
        epoch_losses = []
        iteration_times = []
        
        for batch_idx, (images, targets) in enumerate(data_loader):
            iter_start_time = time.time()
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            losses.backward()
            optimizer.step()
            
            # Loss値を記録
            epoch_losses.append({k: v.item() for k, v in loss_dict.items()})
            
            iter_time = time.time() - iter_start_time
            iteration_times.append(iter_time)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {losses.item():.4f}, Time = {iter_time:.2f}s")
        
        # update the learning rate
        lr_scheduler.step()
        
        # エポック終了後の処理
        epoch_time = time.time() - epoch_start_time
        
        # メモリ使用量を記録
        if torch.cuda.is_available():
            gpu_memory_after = torch.cuda.memory_allocated(device) / 1024**2  # MB
        cpu_memory_after = psutil.virtual_memory().used / 1024**2  # MB
        
        # 平均loss値を計算
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = np.mean([loss[key] for loss in epoch_losses])
        
        # Loss履歴に追加
        loss_history['epoch'].append(epoch)
        
        # 総Lossを計算
        total_loss = sum(avg_losses.values())
        if 'total_loss' not in loss_history:
            loss_history['total_loss'] = []
        loss_history['total_loss'].append(total_loss)
        
        # 各Loss成分を記録
        for loss_type, loss_value in avg_losses.items():
            if loss_type not in loss_history:
                loss_history[loss_type] = []
            loss_history[loss_type].append(loss_value)
        
        # パフォーマンス情報を記録
        performance_info = {
            'epoch': epoch,
            'epoch_time': epoch_time,
            'avg_iteration_time': np.mean(iteration_times),
            'cpu_memory_before': cpu_memory_before,
            'cpu_memory_after': cpu_memory_after,
        }
        
        if torch.cuda.is_available():
            performance_info.update({
                'gpu_memory_before': gpu_memory_before,
                'gpu_memory_after': gpu_memory_after,
            })
        
        performance_log.append(performance_info)
        
        print(f"Epoch {epoch} 完了: 時間={epoch_time:.2f}s, 平均iteration時間={np.mean(iteration_times):.3f}s")
        
        # evaluate on the test dataset
        pt.evaluate(model, data_loader_test, device=device)

    overall_training_time = time.time() - overall_start_time
    
    print("学習完了!")
    print(f"全体の学習時間: {overall_training_time:.2f}秒")
    
    # 学習後の評価
    print("学習後の評価を実行中...")
    post_training_eval = evaluate_model(model, data_loader_test, device)
    post_training_vis = visualize_predictions(model, dataset_test, device, num_samples=5, save_dir='images/finetuning/post_training')
    
    # Loss曲線をプロット
    plot_loss_curves(loss_history)
    
    # 結果をまとめて保存
    results = {
        'training_config': training_config,
        'training_time': overall_training_time,
        'loss_history': loss_history,
        'performance_log': performance_log,
        'pre_training_evaluation': pre_training_eval,
        'post_training_evaluation': post_training_eval,
        'pre_training_visualization': pre_training_vis,
        'post_training_visualization': post_training_vis
    }
    
    with open('training_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print("結果がtraining_results.jsonに保存されました")
    print("可視化画像がimagesディレクトリに保存されました")

if __name__ == "__main__":
    main()