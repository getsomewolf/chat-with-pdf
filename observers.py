from event_manager import Observer # Assuming event_manager.py is in the same root
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class LoggingObserver(Observer):
    def update(self, event_type: str, data: dict = None):
        log_message = f"[EVENT] {event_type}"
        if data:
            log_message += f": {data}"
        logger.info(log_message)

class CLILoggingObserver(Observer): # Specific for CLI if more detailed CLI output is needed
    def update(self, event_type: str, data: dict = None):
        print(f"[CLI EVENT] {event_type}: {data if data else ''}")

# Example of another observer if needed in the future
# class MetricsObserver(Observer):
#     def update(self, event_type: str, data: dict = None):
#         if event_type == "generation_completed" and data and "time" in data:
#             # Send to a metrics system, e.g., Prometheus, StatsD
#             logger.info(f"[METRIC] LLM Generation Time: {data['time']:.2f}s")
