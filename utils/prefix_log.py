from utils.misc import get_timestamp


class PrefixLog:
    def __init__(self, prefix=None):
        self.prefix = prefix

    def log(self, txt):
        prefix = ""
        if self.prefix:
            prefix += f"[{self.prefix}] "
        print(f"{prefix}{txt}")


class PrefixLogMixin:
    def log(self, txt):
        cls_name = self.__class__.__name__
        print(f"[{cls_name}] {txt}")


class TimeLogMixin:
    def log(self, txt):
        cls_name = self.__class__.__name__
        print(f"[{get_timestamp(format='time')}][{cls_name}] {txt}")
