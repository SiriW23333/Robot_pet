import datetime
import json
import os
from typing import Dict, Any

class Timer:
    """ÇáÁ¿¼¶¼ÆÊ±Æ÷£¨ÒÆ³ýPyTorchÒÀÀµ£©"""
    def __init__(self):
        self.clock: Dict[str, datetime.datetime] = {}

    def start(self, key: str = "default") -> None:
        """¿ªÊ¼¼ÆÊ±"""
        self.clock[key] = datetime.datetime.now()

    def end(self, key: str = "default") -> float:
        """½áÊø¼ÆÊ±²¢·µ»ØÃëÊý"""
        if key not in self.clock:
            raise KeyError(f"Timer key '{key}' not found")
        interval = datetime.datetime.now() - self.clock[key]
        del self.clock[key]
        return interval.total_seconds()

def str2bool(s: str) -> bool:
    """×Ö·û´®×ª²¼¶ûÖµ£¨¼æÈÝµØ¹ÏRDKµÄÅäÖÃ¶ÁÈ¡£©"""
    return str(s).lower() in ('true', '1', 'yes', 'y', 't')

def save_checkpoint(
    epoch: int,
    model_state: bytes,
    best_score: float,
    checkpoint_path: str,
    model_path: str
) -> None:
    """
    ±£´æ¼ì²éµã£¨ÊÊÅäµØÆ½ÏßRDKµÄ¶þ½øÖÆ¸ñÊ½£©
    
    ²ÎÊý:
        epoch: µ±Ç°ÑµÁ·ÂÖ´Î
        model_state: Ä£ÐÍ×´Ì¬£¨¶þ½øÖÆ¸ñÊ½£©
        best_score: ×î¼Ñ¾«¶ÈÖµ
        checkpoint_path: ¼ì²éµã±£´æÂ·¾¶
        model_path: Ä£ÐÍ±£´æÂ·¾¶
    """
    checkpoint = {
        'epoch': epoch,
        'model_state': model_state,
        'best_score': best_score,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    # ±£´æ¼ì²éµã£¨JSON¸ñÊ½£©
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f)
    
    # ±£´æÄ£ÐÍ£¨¶þ½øÖÆ¸ñÊ½£©
    with open(model_path, 'wb') as f:
        f.write(model_state)

def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """¼ÓÔØ¼ì²éµã£¨ÊÊÅäµØÆ½ÏßRDK£©"""
    with open(checkpoint_path, 'r') as f:
        data = json.load(f)
    
    # ½«×Ö·û´®Ê±¼ä×ª»Ødatetime¶ÔÏó
    if 'timestamp' in data:
        data['timestamp'] = datetime.datetime.fromisoformat(data['timestamp'])
    return data

def store_labels(path: str, labels: list) -> None:
    """´æ´¢±êÇ©ÎÄ¼þ£¨¼æÈÝRDKµÄÎÄ¼þÏµÍ³£©"""
    # È·±£Ä¿Â¼´æÔÚ
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Ð´Èë±êÇ©£¨Ã¿ÐÐÒ»¸ö£©
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(str(label) for label in labels))