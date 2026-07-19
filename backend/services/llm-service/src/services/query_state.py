from enum import Enum
from typing import Dict


class QueryState(Enum):
    """Query analysis states."""
    SHORT = "short"          
    WELL_FORMED = "well_formed"    
    LONG = "long"            
    
    @staticmethod
    def get_thresholds() -> Dict[str, int]:
        return {
            "short_threshold": 40,
            "long_threshold": 70
        }
    
    @classmethod
    def analyze_query_state(cls, query: str):
        length = len(query.strip())
        thresholds = cls.get_thresholds()
        
        if length < thresholds["short_threshold"]:
            return cls.SHORT
        elif length <= thresholds["long_threshold"]:
            return cls.WELL_FORMED
        else:
            return cls.LONG
