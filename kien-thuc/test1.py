import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
from scipy.stats import entropy as scipy_entropy

# Hàm tính entropy (entropy của phân phối nhãn)
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return scipy_entropy(probs, base=2)

# Hàm tính Information Gain (giữ nguyên)
def info_gain(y, left_idx, right_idx):
    H_parent = entropy(y)
    left_y, right_y = y[left_idx], y[right_idx]
    w_left, w_right = len(left_y) / len(y), len(right_y) / len(y)
    return H_parent - (w_left * entropy(left_y) + w_right * entropy(right_y))

# Hàm vẽ sơ đồ trực quan hóa
def visualize_info_gain(y, left_idx, right_idx):
    left_y, right_y = y[left_idx], y[right_idx]
    
    # Tính toán entropy
    H_parent = entropy(y)
    H_left = entropy(left_y)
    H_right = entropy(right_y)
    w_left = len(left_y) / len(y)
    w_right = len(right_y) / len(y)
    IG = info_gain(y, left_idx, right_idx)
    
    print(f"H(parent) = {H_parent:.3f}")
    print(f"H(left) = {H_left:.3f} (weight={w_left:.2f})")
    print(f"H(right) = {H_right:.3f} (weight={w_right:.2f})")
    print(f"Information Gain = {IG:.3f}")
    
    # Vẽ sơ đồ flow bằng graphviz
    dot = Digraph()
    dot.node('P', f'Parent\nH={H_parent:.3f}\nn={len(y)}')
    dot.node('L', f'Left\nH={H_left:.3f}\nn={len(left_y)}\nw={w_left:.2f}')
    dot.node('R', f'Right\nH={H_right:.3f}\nn={len(right_y)}\nw={w_right:.2f}')
    dot.edges([('P', 'L'), ('P', 'R')])
    
    # Save graphviz visualization to file instead of using display
    dot.render('decision_tree_split', format='png', cleanup=True)
    print("Graphviz visualization saved as 'decision_tree_split.png'")
    
    # Vẽ biểu đồ bar entropy
    labels = ['Parent', 'Left', 'Right']
    entropies = [H_parent, H_left, H_right]
    weights = [1, w_left, w_right]
    
    fig, ax = plt.subplots(figsize=(6,4))
    bars = ax.bar(labels, entropies, color=['blue', 'green', 'orange'], alpha=0.7)
    
    for bar, w in zip(bars, weights):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.05, f'w={w:.2f}', ha='center')
    
    ax.set_ylabel('Entropy (bits)')
    ax.set_title('Entropy of Parent and Children Splits')
    plt.ylim(0, max(entropies)+0.5)
    plt.show()

# Example usage
# Tập nhãn giả lập
y = np.array([0,0,1,1,1,0,1,0])  
left_idx = np.array([0,1,5,7])    # indexes of left split
right_idx = np.array([2,3,4,6])   # indexes of right split

visualize_info_gain(y, left_idx, right_idx)
