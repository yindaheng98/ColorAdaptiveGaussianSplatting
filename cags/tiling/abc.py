import abc
from typing import List

from gaussian_splatting.gaussian_model import GaussianModel


class AbstractTiling(abc.ABC):
    @abc.abstractmethod
    def tiling(self, model: GaussianModel) -> List[GaussianModel]:
        raise NotImplementedError

    @abc.abstractmethod
    def stitching(self, models: List[GaussianModel]) -> GaussianModel:
        raise NotImplementedError
