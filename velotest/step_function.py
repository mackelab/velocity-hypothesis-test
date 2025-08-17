from functools import lru_cache

import torch
import numpy as np


class StepFunction1D:
    __ranges: torch.Tensor
    __values: torch.Tensor

    def __init__(self, ranges: torch.Tensor, values: torch.Tensor):
        """

        :param ranges:  A 2D tensor of shape (n_ranges, 2) where each row is a range [start, end]. Expressed in radians.
            We assume:
            - Value of ranges are normalised to visualised velocity such that 0 is the position of the visualised velocity.
            - No overlap between ranges.
            - Ranges are sorted in ascending order.
            - Ranges are in [0,2*pi]. No range is spanning from below 2*pi to above 0.
        :param values:
        """
        assert ranges.shape[0] == values.shape[0], "Ranges and values must have the same number of elements"
        assert ranges.shape[1] == 2, "Ranges must be a 2D tensor with shape (n_ranges, 2)"
        # Assert start of range sorted in ascending order
        assert torch.all(ranges[:-1, 0] <= ranges[1:, 0]), "Ranges are not sorted in ascending order"
        # Assert no overlap between ranges
        for i in range(ranges.shape[0] - 1):
            assert ranges[i, 1] <= ranges[i + 1, 0], "Ranges overlap"
        # Assert no range is spanning from below 2*pi to above 0
        assert torch.all(ranges[:, 0] >= 0) and torch.all(ranges[:, 1] <= 2 * np.pi), "Ranges are not in [0, 2*pi]"
        self.__ranges = ranges.to(torch.float)
        self.__values = values.to(torch.float)

    @lru_cache(maxsize=5)
    def subset_function(self, exclusion_angle: float = None):
        """

        :param exclusion_angle: Angle around visualised velocity position where we ignore samples [in radians].
        :return:
        """
        if exclusion_angle is not None:
            ranges = self.__ranges.clone()
            values = self.__values.clone()
            # Iterating through tensor from the beginning
            for start, end in self.__ranges:
                if end < exclusion_angle:
                    # Remove range if it is completely before the exclusion angle
                    ranges = ranges[1:]
                    values = values[1:]
                elif start < exclusion_angle < end:
                    # Adjust the range to exclude the angle
                    ranges[ranges[:, 0] == start, 0] = exclusion_angle
                else:
                    # No adjustment needed
                    break
            # Iterating through tensor from the end
            for start, end in torch.flip(self.__ranges, dims=(0,)):
                if start > 2 * np.pi - exclusion_angle:
                    # Remove range if it is completely after the exclusion angle
                    ranges = ranges[:-1]
                    values = values[:-1]
                elif start < 2 * np.pi - exclusion_angle < end:
                    # Adjust the range to exclude the angle
                    ranges[ranges[:, 1] == end, 1] = 2 * np.pi - exclusion_angle
                else:
                    # No adjustment needed
                    break
            return ranges, values
        else:
            return self.__ranges, self.__values

    def get_ranges(self, exclusion_angle: float = None):
        ranges = self.subset_function(exclusion_angle)[0]
        return ranges

    def get_values(self, exclusion_angle: float = None):
        return self.subset_function(exclusion_angle)[1]

    def __call__(self, x: torch.Tensor, exclusion_angle: float = None):
        """

        :param x:
        :param exclusion_angle: Angle around visualised velocity position where we ignore samples [in radians].
        :return:
        """
        assert x.ndim == 1
        result = torch.ones_like(x, dtype=self.get_values(exclusion_angle).dtype)
        result *= torch.nan  # Initialize with NaN to indicate no value assigned
        for i in range(self.get_ranges(exclusion_angle).shape[0]):
            mask = (x >= self.get_ranges(exclusion_angle)[i, 0]) & (x <= self.get_ranges(exclusion_angle)[i, 1])
            result[mask] = self.get_values(exclusion_angle)[i]
        # Check that all x were in domain
        assert not torch.any(torch.isnan(result)), "X is not (completely) in the domain of the step function"
        return result

    def get_domain(self, exclusion_angle: float = None):
        """

        :param exclusion_angle: Angle around visualised velocity position where we ignore samples [in radians].
        :return:
        """
        domain = []
        start_domain = self.get_ranges(exclusion_angle)[0, 0]
        end_domain = self.get_ranges(exclusion_angle)[0, 1]
        for i in range(1, self.get_ranges(exclusion_angle).shape[0]):
            if self.get_ranges(exclusion_angle)[i, 0] > end_domain:
                domain.append((start_domain, end_domain))
                start_domain = self.get_ranges(exclusion_angle)[i, 0]
            end_domain = self.get_ranges(exclusion_angle)[i, 1]
        domain.append((start_domain, end_domain))
        return domain
