import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as mtransforms  # 导入变换模块

plt.rcParams['font.family'] = 'Arial'

labels = ['AUC', 'AUPR', 'Recall', 'Acc']
num_labels = len(labels)

angles = np.linspace(0, 2 * np.pi, num_labels, endpoint=False).tolist()
angles += angles[:1]

datasets = {
    'Celegans': {
        'FLGDTI_without_ECA': [0.993, 0.994, 0.958, 0.963],
        'FLG-DTI': [0.994, 0.995, 0.969, 0.969]
    },
    'Human': {
        'FLGDTI_without_ECA': [0.984, 0.984, 0.916, 0.932],
        'FLG-DTI': [0.989, 0.989, 0.931, 0.940]
    },
    'BioSNAP': {
        'FLGDTI_without_ECA': [0.841, 0.845, 0.753, 0.766],
        'FLG-DTI': [0.852, 0.855, 0.772, 0.782]
    }
}

fig, axs = plt.subplots(1, 3, subplot_kw=dict(polar=True), figsize=(18, 6))

colors = {
    'FLGDTI_without_ECA': 'tomato',
    'FLG-DTI': 'cornflowerblue'
}

axis_ranges = {
    'Celegans': (0.9, 1.0),
    'Human': (0.9, 1.0),
    'BioSNAP': (0.74, 0.86)
}

for ax, (dataset, methods) in zip(axs, datasets.items()):
    for method, values in methods.items():
        values += values[:1]
        ax.plot(angles, values, label=method, color=colors[method], linewidth=2)
        ax.fill(angles, values, color=colors[method], alpha=0.25)

    ax.set_title(dataset, size=16)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(axis_ranges[dataset])
    ax.tick_params(labelsize=14)

# 调整AUPR标签的位置
for ax in axs:
    for label in ax.get_xticklabels():
        if label.get_text() == 'AUPR':
            tiny_scale = 0.99  # 缩放因子（可根据需要调整）
            offset = (
                mtransforms.Affine2D()
                .scale(tiny_scale, 1)  # 在x方向缩小坐标系
                .translate(1, 0)       # 在缩放后的坐标系中移动1单位
            )
            # 应用组合变换
            label.set_transform(label.get_transform() + offset)
            # 优化对齐方式
            label.set_horizontalalignment('left')
            label.set_verticalalignment('center_baseline')
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=2, fontsize=14)

plt.tight_layout(rect=[0, 0.15, 1, 0.95])
plt.savefig(r"C:\Users\Y小鬼H\Desktop\radar_chart.png", dpi=300, bbox_inches='tight')
plt.show()