from abc import ABC
from abc import abstractmethod


class MetricABC(ABC):
    """Metric base class.

    Subclasses of the MetricABC must implement the __call__ method so that they can be callable by evaluation
    procedures.
    To get the name of the metric, access the `name` attribute, which gets set during the initialization method of the
    base class, to the name of the subclass name.
    """
    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
