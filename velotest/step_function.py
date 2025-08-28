from functools import lru_cache

import numpy as np


class StepFunction1D:
    __ranges: np.ndarray
    __values: np.ndarray
    __offset: float

    def __init__(self, ranges: np.ndarray, values: np.ndarray, offset: float):
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
        assert np.all(ranges[:-1, 0] <= ranges[1:, 0]), "Ranges are not sorted in ascending order"
        # Assert no overlap between ranges
        for i in range(ranges.shape[0] - 1):
            assert ranges[i, 1] <= ranges[i + 1, 0], "Ranges overlap"
        # Assert no range is spanning from below 2*pi to above 0
        assert np.all(ranges[:, 0] >= 0) and np.all(ranges[:, 1] <= 2 * np.pi), "Ranges are not in [0, 2*pi]"
        self.__ranges = ranges.astype(np.float32)
        self.__values = values.astype(np.float32)
        self.__offset = offset

    @lru_cache(maxsize=5)
    def subset_function(self, exclusion_angle: float = None):
        """

        :param exclusion_angle: Angle around visualised velocity position where we ignore samples [in radians].
        :return:
        """
        if exclusion_angle is not None:
            ranges = self.__ranges.copy()
            values = self.__values.copy()
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
            for start, end in np.flip(self.__ranges, axis=(0,)):
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

    def get_ranges(self, exclusion_angle: float = None, offset: bool = False):
        if not offset:
            return self.subset_function(exclusion_angle)[0]
        else:
            return self.subset_function(exclusion_angle)[0] + self.__offset

    def get_values(self, exclusion_angle: float = None):
        return self.subset_function(exclusion_angle)[1]

    def __call__(self, x: np.ndarray, exclusion_angle: float = None):
        """

        :param x:
        :param exclusion_angle: Angle around visualised velocity position where we ignore samples [in radians].
        :return:
        """
        assert x.ndim == 1
        result = np.ones_like(x, dtype=self.get_values(exclusion_angle).dtype)
        result *= np.nan  # Initialize with NaN to indicate no value assigned
        for i in range(self.get_ranges(exclusion_angle).shape[0]):
            mask = (x >= self.get_ranges(exclusion_angle)[i, 0]) & (x <= self.get_ranges(exclusion_angle)[i, 1])
            result[mask] = self.get_values(exclusion_angle)[i]
        # Check that all x were in domain
        assert not np.any(np.isnan(result)), "X is not (completely) in the domain of the step function"
        return result

    def get_domain(self, exclusion_angle: float = None, offset: bool = False):
        """

        :param exclusion_angle: Angle around visualised velocity position where we ignore samples [in radians].
        :return:
        """
        domain = []
        ranges = self.get_ranges(exclusion_angle, offset=offset)
        start_domain = ranges[0, 0]
        end_domain = ranges[0, 1]
        for i in range(1, ranges.shape[0]):
            if ranges[i, 0] > end_domain:
                domain.append((start_domain, end_domain))
                start_domain = ranges[i, 0]
            end_domain = ranges[i, 1]
        domain.append((start_domain, end_domain))
        return domain

    def get_max_value(self, exclusion_angle: float = None, offset: bool = False):
        """
        Returns the range of the maximum value and the value itself of the step function.
        :param exclusion_angle: Angle around visualised velocity position where we ignore samples [in radians].
        :param offset: If True, converts the maximum value back to the original direction.
        :return: Range in which maximum value occurs and value as a float.
        """
        values = self.get_values(exclusion_angle)
        max_index = np.argmax(values)
        return self.get_ranges(exclusion_angle, offset=offset)[max_index], values[max_index]

    def sample_from(self, domain, sampling_stepsize: float = 0.0001):
        domain_samples = []
        for start, end in domain:
            domain_samples.append(np.arange(start, end, sampling_stepsize))
        x = np.concatenate(domain_samples)
        return x, self(x)
