import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 日本語フォントの設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

def create_improved_graphs_from_results():
    """training_results.jsonから実際のデータを読み込んでY軸を0から始めるグラフを作成"""
    
    # 結果ファイルが存在するかチェック
    if not os.path.exists('training_results.json'):
        print("training_results.jsonが見つかりません。学習が完了していない可能性があります。")
        return
    
    # 結果データを読み込み
    with open('training_results.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    loss_history = results.get('loss_history', {})
    performance_log = results.get('performance_log', [])
    
    if not loss_history.get('epoch') or not performance_log:
        print("学習データが不完全です。")
        return
    
    epochs = np.array(loss_history['epoch'])
    print(f"読み込んだエポック数: {len(epochs)}")
    print(f"利用可能なLoss成分: {[k for k in loss_history.keys() if k != 'epoch']}")
    
    # 1. Loss推移グラフ（Y軸0から開始）
    plt.figure(figsize=(14, 10))
    
    # 総Loss推移
    if 'total_loss' in loss_history:
        total_loss = np.array(loss_history['total_loss'])
        plt.subplot(2, 2, 1)
        plt.plot(epochs, total_loss, 'b-', linewidth=3, marker='o', markersize=8, label='Total Loss')
        plt.xlabel('エポック', fontsize=12)
        plt.ylabel('Total Loss値', fontsize=12)
        plt.title('総Loss推移 (実データ)', fontsize=14, fontweight='bold')
        plt.ylim(0, max(total_loss) * 1.1)  # 0から開始、最大値の1.1倍まで
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    # 個別Loss成分（Y軸0から開始）
    plt.subplot(2, 2, 2)
    colors = ['red', 'green', 'purple', 'orange', 'brown']
    markers = ['s', '^', 'd', 'v', '<']
    
    max_loss_value = 0
    loss_components = ['loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg']
    
    for i, (component, color, marker) in enumerate(zip(loss_components, colors, markers)):
        if component in loss_history and len(loss_history[component]) > 0:
            values = np.array(loss_history[component])
            max_loss_value = max(max_loss_value, max(values))
            plt.plot(epochs, values, color=color, linestyle='--', linewidth=2, 
                    marker=marker, markersize=6, label=component.replace('loss_', '').replace('_', ' ').title())
    
    plt.xlabel('エポック', fontsize=12)
    plt.ylabel('Loss値', fontsize=12)
    plt.title('Loss成分別推移 (実データ)', fontsize=14, fontweight='bold')
    plt.ylim(0, max_loss_value * 1.1)  # 0から開始
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # 実行時間推移（Y軸0から開始）
    plt.subplot(2, 2, 3)
    epoch_times = [log.get('epoch_time', 0) for log in performance_log]
    
    if epoch_times:
        plt.plot(epochs, epoch_times, 'b-', linewidth=3, marker='o', markersize=8, label='エポック実行時間')
        plt.xlabel('エポック', fontsize=12)
        plt.ylabel('時間 (秒)', fontsize=12)
        plt.title('実行時間推移 (実データ)', fontsize=14, fontweight='bold')
        plt.ylim(0, max(epoch_times) * 1.1)  # 0から開始
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # メモリ使用量推移（Y軸0から開始）
    plt.subplot(2, 2, 4)
    cpu_memory_after = [log.get('cpu_memory_after', 0) for log in performance_log]
    
    max_memory = max(cpu_memory_after) if cpu_memory_after else 1000
    
    if cpu_memory_after:
        plt.plot(epochs, cpu_memory_after, 'g-', linewidth=2, marker='^', markersize=6, label='CPU メモリ')
    
    # GPUメモリがある場合
    if performance_log and 'gpu_memory_after' in performance_log[0]:
        gpu_memory_after = [log.get('gpu_memory_after', 0) for log in performance_log]
        if gpu_memory_after:
            plt.plot(epochs, gpu_memory_after, 'r-', linewidth=3, marker='s', markersize=8, label='GPU メモリ')
            max_memory = max(max_memory, max(gpu_memory_after))
    
    plt.xlabel('エポック', fontsize=12)
    plt.ylabel('メモリ使用量 (MB)', fontsize=12)
    plt.title('メモリ使用量推移 (実データ)', fontsize=14, fontweight='bold')
    plt.ylim(0, max_memory * 1.1)  # 0から開始
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Mask R-CNN ファインチューニング - 実際の学習データ分析 (Y軸0基準)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/finetuning/improved_training_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Loss推移の詳細グラフ（Y軸0から開始）
    plt.figure(figsize=(12, 8))
    if 'total_loss' in loss_history:
        total_loss = np.array(loss_history['total_loss'])
        plt.plot(epochs, total_loss, 'b-', linewidth=4, marker='o', markersize=10, label='Total Loss', alpha=0.8)
        max_loss_for_detail = max(total_loss)
    else:
        max_loss_for_detail = 0
    
    for i, (component, color, marker) in enumerate(zip(loss_components, colors, markers)):
        if component in loss_history and len(loss_history[component]) > 0:
            values = np.array(loss_history[component])
            max_loss_for_detail = max(max_loss_for_detail, max(values))
            plt.plot(epochs, values, color=color, linestyle='--', linewidth=2,
                    marker=marker, markersize=7, label=component.replace('loss_', '').replace('_', ' ').title())
    
    plt.xlabel('エポック', fontsize=14)
    plt.ylabel('Loss値', fontsize=14)
    plt.title('実際の学習におけるLoss推移（各成分詳細、Y軸0基準）', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.5, len(epochs) - 0.5)
    plt.ylim(0, max_loss_for_detail * 1.1)  # 0から開始
    plt.tight_layout()
    plt.savefig('images/finetuning/improved_loss_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. パフォーマンス詳細グラフ（Y軸0から開始）
    plt.figure(figsize=(12, 6))
    
    # 実行時間
    plt.subplot(1, 2, 1)
    if epoch_times:
        plt.plot(epochs, epoch_times, 'b-', linewidth=3, marker='o', markersize=8)
        plt.fill_between(epochs, 0, epoch_times, alpha=0.2, color='blue')
        plt.xlabel('エポック', fontsize=12)
        plt.ylabel('実行時間 (秒)', fontsize=12)
        plt.title('エポック実行時間推移 (実データ、Y軸0基準)', fontsize=14, fontweight='bold')
        plt.ylim(0, max(epoch_times) * 1.1)  # 0から開始
        plt.grid(True, alpha=0.3)
    
    # iteration時間推移も追加
    avg_iter_times = [log.get('avg_iteration_time', 0) for log in performance_log]
    if avg_iter_times:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, avg_iter_times, 'purple', linewidth=3, marker='d', markersize=8, label='平均iteration時間')
        plt.fill_between(epochs, 0, avg_iter_times, alpha=0.2, color='purple')
        plt.xlabel('エポック', fontsize=12)
        plt.ylabel('平均iteration時間 (秒)', fontsize=12)
        plt.title('iteration時間推移 (実データ、Y軸0基準)', fontsize=14, fontweight='bold')
        plt.ylim(0, max(avg_iter_times) * 1.1)  # 0から開始
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/finetuning/improved_performance_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 統計情報を計算
    stats = {}
    if total_loss is not None and len(total_loss) > 0:
        stats.update({
            'epochs': len(epochs),
            'initial_total_loss': float(total_loss[0]),
            'final_total_loss': float(total_loss[-1]),
            'loss_reduction_rate': float((total_loss[0] - total_loss[-1]) / total_loss[0] * 100),
        })
    
    if epoch_times:
        stats.update({
            'avg_epoch_time': float(np.mean(epoch_times)),
            'total_training_time': float(np.sum(epoch_times)),
        })
    
    if avg_iter_times:
        stats['avg_iteration_time'] = float(np.mean(avg_iter_times))
    
    if performance_log and 'gpu_memory_after' in performance_log[0]:
        gpu_memory_final = performance_log[-1].get('gpu_memory_after', 0)
        stats['final_gpu_memory'] = float(gpu_memory_final)
    
    print("実際のデータから以下のグラフを作成しました（Y軸0基準）:")
    print("- images/finetuning/improved_training_comprehensive.png (総合分析)")
    print("- images/finetuning/improved_loss_detailed.png (Loss推移詳細)")
    print("- images/finetuning/improved_performance_detailed.png (パフォーマンス詳細)")
    print()
    print("学習統計 (実測値):")
    for key, value in stats.items():
        if 'time' in key.lower():
            print(f"- {key}: {value:.2f}秒" if isinstance(value, float) else f"- {key}: {value}")
        elif 'rate' in key.lower():
            print(f"- {key}: {value:.1f}%")
        elif 'memory' in key.lower():
            print(f"- {key}: {value:.0f}MB")
        elif 'loss' in key.lower():
            print(f"- {key}: {value:.4f}")
        else:
            print(f"- {key}: {value}")
    
    return stats

if __name__ == "__main__":
    stats = create_improved_graphs_from_results()
    print("\n✅ Y軸0基準の改良されたグラフが作成されました！") 