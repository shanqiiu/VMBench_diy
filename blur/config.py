# -*- coding: utf-8 -*-
"""
��Ƶģ����������ļ�
"""

import os
from pathlib import Path

class BlurDetectionConfig:
    """ģ�����������"""
    
    def __init__(self):
        """��ʼ������"""
        # ����·������
        self.base_dir = Path(__file__).parent.parent
        self.cache_dir = self.base_dir / ".cache"
        self.output_dir = self.base_dir / "��Ƶģ�����" / "results"
        
        # ģ��·������
        self.model_paths = {
            'q_align_model': str(self.cache_dir / "q-future" / "one-align"),
            'grounding_dino_config': str(self.base_dir / "Grounded-Segment-Anything" / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinB.py"),
            'grounding_dino_checkpoint': str(self.cache_dir / "groundingdino_swinb_cogcoor.pth"),
            'bert_path': str(self.cache_dir / "google-bert" / "bert-base-uncased"),
            'sam_checkpoint': str(self.cache_dir / "sam_vit_h_4b8939.pth"),
            'cotracker_checkpoint': str(self.cache_dir / "scaled_offline.pth")
        }
        
        # ����������
        self.detection_params = {
            'window_size': 3,  # �������ڴ�С
            'blur_thresholds': {
                'mild_blur': 0.015,    # ��΢ģ����ֵ
                'moderate_blur': 0.025, # �е�ģ����ֵ
                'severe_blur': 0.04    # ����ģ����ֵ
            },
            'confidence_threshold': 0.7,  # �ۺ����Ŷ���ֵ
            'min_frames': 10,  # ��С֡��Ҫ��
            'max_frames': 1000  # ���֡������
        }
        
        # ���ӻ�����
        self.visualization_params = {
            'figure_size': (15, 10),
            'dpi': 300,
            'font_size': 12,
            'color_palette': 'husl',
            'style': 'whitegrid'
        }
        
        # �������
        self.output_params = {
            'save_visualizations': True,
            'save_detailed_reports': True,
            'save_csv_summary': True,
            'save_json_results': True
        }
        
        # �豸����
        self.device_config = {
            'device': 'cuda:0',
            'batch_size': 1,
            'num_workers': 4
        }
        
        # ȷ�����Ŀ¼����
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_name: str) -> str:
        """��ȡģ��·��"""
        return self.model_paths.get(model_name, "")
    
    def get_detection_param(self, param_name: str):
        """��ȡ������"""
        return self.detection_params.get(param_name)
    
    def get_visualization_param(self, param_name: str):
        """��ȡ���ӻ�����"""
        return self.visualization_params.get(param_name)
    
    def get_output_param(self, param_name: str):
        """��ȡ�������"""
        return self.output_params.get(param_name)
    
    def get_device_config(self, config_name: str):
        """��ȡ�豸����"""
        return self.device_config.get(config_name)
    
    def update_model_path(self, model_name: str, new_path: str):
        """����ģ��·��"""
        self.model_paths[model_name] = new_path
    
    def update_detection_param(self, param_name: str, new_value):
        """���¼�����"""
        self.detection_params[param_name] = new_value
    
    def update_visualization_param(self, param_name: str, new_value):
        """���¿��ӻ�����"""
        self.visualization_params[param_name] = new_value
    
    def update_output_param(self, param_name: str, new_value):
        """�����������"""
        self.output_params[param_name] = new_value
    
    def update_device_config(self, config_name: str, new_value):
        """�����豸����"""
        self.device_config[config_name] = new_value
    
    def validate_config(self) -> bool:
        """��֤�����Ƿ���Ч"""
        # ����Ҫ��ģ���ļ��Ƿ����
        required_models = ['q_align_model', 'grounding_dino_checkpoint', 'sam_checkpoint', 'cotracker_checkpoint']
        
        for model in required_models:
            model_path = self.get_model_path(model)
            if not os.path.exists(model_path):
                print(f"����: ģ���ļ������� - {model}: {model_path}")
                return False
        
        # ������Ŀ¼�Ƿ��д
        if not os.access(self.output_dir, os.W_OK):
            print(f"����: ���Ŀ¼����д - {self.output_dir}")
            return False
        
        return True
    
    def print_config(self):
        """��ӡ��ǰ����"""
        print("=== ��Ƶģ��������� ===")
        print(f"����Ŀ¼: {self.base_dir}")
        print(f"����Ŀ¼: {self.cache_dir}")
        print(f"���Ŀ¼: {self.output_dir}")
        print("\nģ��·��:")
        for name, path in self.model_paths.items():
            exists = "?" if os.path.exists(path) else "?"
            print(f"  {name}: {path} {exists}")
        print("\n������:")
        for name, value in self.detection_params.items():
            print(f"  {name}: {value}")
        print("\n�豸����:")
        for name, value in self.device_config.items():
            print(f"  {name}: {value}")


# Ĭ������ʵ��
default_config = BlurDetectionConfig()

# Ԥ��������
PRESET_CONFIGS = {
    'fast': {
        'window_size': 2,
        'confidence_threshold': 0.6,
        'min_frames': 5
    },
    'accurate': {
        'window_size': 5,
        'confidence_threshold': 0.8,
        'min_frames': 20
    },
    'balanced': {
        'window_size': 3,
        'confidence_threshold': 0.7,
        'min_frames': 10
    }
}

def get_preset_config(preset_name: str) -> BlurDetectionConfig:
    """��ȡԤ��������"""
    config = BlurDetectionConfig()
    
    if preset_name in PRESET_CONFIGS:
        preset_params = PRESET_CONFIGS[preset_name]
        for param, value in preset_params.items():
            config.update_detection_param(param, value)
    
    return config

def create_custom_config(**kwargs) -> BlurDetectionConfig:
    """�����Զ�������"""
    config = BlurDetectionConfig()
    
    # ���¼�����
    if 'detection_params' in kwargs:
        for param, value in kwargs['detection_params'].items():
            config.update_detection_param(param, value)
    
    # ����ģ��·��
    if 'model_paths' in kwargs:
        for model, path in kwargs['model_paths'].items():
            config.update_model_path(model, path)
    
    # �����豸����
    if 'device_config' in kwargs:
        for device_param, value in kwargs['device_config'].items():
            config.update_device_config(device_param, value)
    
    return config
