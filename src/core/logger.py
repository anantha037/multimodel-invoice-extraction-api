import logging
import json
import time

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        # Safely extract extra fields injected dynamically
        for key in ["request_id", "latency_ms", "stage", "strategy", "exception", "path", "status", "reason"]:
            if hasattr(record, key):
                log_record[key] = getattr(record, key)
        
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_record)

def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    # Do not propagate to root logger to avoid duplicate console prints
    logger.propagate = False 
    return logger
