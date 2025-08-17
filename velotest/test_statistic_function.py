from velotest.step_function import StepFunction1D


class TestStatistic(StepFunction1D):
    def normalization_factor(self, exclusion_angle: float = None):
        """
        Computes the normalization factor for the step function.
        :param exclusion_angle: Angle around visualised velocity position where we ignore samples [in radians].
        :return: Normalization factor as a float.
        """
        ranges = self.get_ranges(exclusion_angle)
        return (ranges[:, 1] - ranges[:, 0]).sum()

    def p_value(self, t_obs: float, exclusion_angle: float = None):
        """
        Computes the p-value for a given observed test statistic t_obs.
        :param t_obs: Observed test statistic.
        :param exclusion_angle: Angle around visualised velocity position where we ignore samples [in radians].
        :return: p-value as a float.
        """
        ranges = self.get_ranges(exclusion_angle)
        values = self.get_values(exclusion_angle)

        relevant_ranges = ranges[values >= t_obs]
        total_area = (relevant_ranges[:, 1] - relevant_ranges[:, 0]).sum()
        return total_area / self.normalization_factor(exclusion_angle)
