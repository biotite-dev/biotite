# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.database.pubchem"
__author__ = "Patrick Kunzmann"
__all__ = ["ThrottleStatus"]


import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ThrottleStatus:
    """
    This class gives information about the dynamic request throttling
    of *Pubchem*.

    Typically, objects of this class are created from responses of
    the *Pubchem* sever using :meth:`from_response()`, so that
    the the throttle statuses can be read from that object.

    Furthermore, this class provides the :meth:`wait_if_busy()` method,
    that halts the execution for a short time at high loads to ensure
    adherence to *Pubchem* usage policies.

    Parameters
    ----------
    count : float
        A value between 0 and (typically) 1 that indicates the current
        load of this user due to the number of requests.
        If the value exceeds 1, server requests will be blocked.
    time : float
        A value between 0 and (typically) 1 that indicates the current
        load of this user due to the running time of requests.
        If the value exceeds 1, server requests will be blocked.
    service : float
        A value between 0 and (typically) 1 that indicates the current
        general load of the server.
        If the value exceeds 1, the server is overloaded.

    Attributes
    ----------
    count, time, service : float
        Read-only attributes for the parameters given above.
    """

    count: ...
    time: ...
    service: ...

    @staticmethod
    def from_response(response):
        """
        Extract the throttle status from a *Pubchem* server response.

        Parameters
        ----------
        response : requests.Response
            The response from the request to the *Pubchem* server.

        Returns
        -------
        throttle_status : ThrottleStatus
            The extracted throttle status.
        """
        throttle_control = response.headers["X-Throttling-Control"]
        throttle_status = [
            substring.split(")")[0] for substring in throttle_control.split("(")[1:]
        ]
        # Remove '%' sign and convert to int
        count_status, time_status, service_status = [
            int(status[:-1]) for status in throttle_status
        ]
        # Convert from percent
        count_status /= 100
        time_status /= 100
        service_status /= 100
        return ThrottleStatus(count_status, time_status, service_status)

    def wait_if_busy(self, threshold=0.5, wait_time=1.0):
        """
        Halt the execution for a given number of seconds, if the current
        request time or count of this user exceeds the given threshold.

        Parameters
        ----------
        threshold : float, optional
            A value between 0 and 1.
            If the load of either the request time or count exceeds this
            value the execution is halted.
        wait_time : float, optional
            The time in seconds the execution will halt, if the
            threshold is exceeded.
        """
        if self.count > threshold or self.time > threshold:
            time.sleep(wait_time)
