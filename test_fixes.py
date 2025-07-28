#!/usr/bin/env python3
"""
修正されたCareFLTrainingPipelineのテストスクリプト
"""

import yaml
import torch
from carefl.experiments.training_pipeline import CareFLTrainingPipeline
from types import SimpleNamespace

def load_config(config_path='config/config.yaml'):
    """設定ファイルを読み込み"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # 辞書をオブジェクトに変換
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return d
    
    return dict_to_namespace(config_dict)

def test_training_pipeline():
    """修正されたtraining pipelineのテスト"""
    print("🔧 修正されたCareFLTrainingPipelineのテストを開始...")
    
    # 設定を読み込み
    config = load_config()
    
    # より小さなテスト設定
    config.meta_data.n_samples = 100  # サンプル数を減らす
    config.training.epochs = 5  # エポック数を減らす
    config.training.batch_size = 32  # バッチサイズを減らす
    
    print(f"📊 設定: サンプル数={config.meta_data.n_samples}, エポック数={config.training.epochs}")
    
    try:
        # パイプラインを初期化
        pipeline = CareFLTrainingPipeline(config)
        print("✅ パイプライン初期化成功")
        
        # 訓練実行
        print("🚀 訓練開始...")
        flow, loss_vals = pipeline._train()
        print("✅ 訓練完了")
        
        # 結果の確認
        print(f"📈 最終損失: {loss_vals[-1]:.4f}")
        print(f"📉 損失の変化: {loss_vals[0]:.4f} → {loss_vals[-1]:.4f}")
        
        # サンプル生成テスト
        print("🎲 サンプル生成テスト...")
        with torch.no_grad():
            samples = flow.sample(10)
            print(f"✅ サンプル生成成功: shape={samples[-1].shape}")
        
        print("🎉 全てのテストが成功しました！")
        return True
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_pipeline()
    exit(0 if success else 1) 