from abc import ABCMeta, abstractmethod

class IContext(metaclass=ABCMeta):
    pass

class IIntervention(metaclass=ABCMeta):
    pass

class IObservedBehavior(metaclass=ABCMeta):
    pass

class IRecords(metaclass=ABCMeta):
    pass

class IUserBehaviorSimulator(metaclass=ABCMeta):

    @abstractmethod
    def init_state(self) -> None:
        pass

    @abstractmethod
    def next_step(self) -> IContext:
        pass

    @abstractmethod
    def interaction(self, intervention: IIntervention) -> IObservedBehavior:
        pass
   
    @abstractmethod
    def get_records(self) -> IRecords:
        pass

