import numpy as np


def calculate_frechet_inception_distance(embedding_model, real_image, generated_image):
    """_summary_

    Args:
        embedding_model (_type_): _description_
        real_embeddings (_type_): _description_
        generated_embeddings (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    # 埋め込み用モデルでの埋め込み
    real_emb = embedding_model(real_image)
    gen_emb = embedding_model(generated_image)
    
    # 平均と分散を計算
    m = np.mean(real_emb, axis=0)
    m_w = np.mean(gen_emb, axis=0)
    c = np.cov(real_emb.T)
    c_w = np.cov(gen_emb.T)
    
    # ||m - m_w||^2の計算
    
    
    return fid