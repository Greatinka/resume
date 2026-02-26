from config import IS_LMS_ENV, MODEL_PATH_LMS


def get_model_path(path: str) -> str:
    """
    Determine the model file path based on the environment.
    
    Adapts paths for both local environment and work environment (LMS).
    
    Args:
        path: Default model path for local environment
        
    Returns:
        Correct model path based on current environment
    """
    if IS_LMS_ENV:
        return MODEL_PATH_LMS
    return path
