from config import IS_LMS_ENV, MODEL_PATH_LMS


def get_model_path(path: str) -> str:
    """Determine the model file path based on the environment"""
    
    if IS_LMS_ENV:
        return MODEL_PATH_LMS
    return path
