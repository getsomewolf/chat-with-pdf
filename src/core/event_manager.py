from abc import ABC, abstractmethod

class Observer(ABC):
    @abstractmethod
    def update(self, event_type: str, data: dict = None):
        pass

class EventManager:
    def __init__(self):
        self.listeners = {}

    def subscribe(self, event_type: str, listener: Observer):
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(listener)

    def emit(self, event_type: str, data: dict = None):
        for listener in self.listeners.get(event_type, []):
            listener.update(event_type, data)