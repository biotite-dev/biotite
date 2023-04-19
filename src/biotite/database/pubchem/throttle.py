# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.database.pubchem"
__author__ = "Patrick Kunzmann"
__all__ = ["ThrottleStatus"]


import time


class ThrottleStatus:

    def __init__(self, count, time, service):
        self._count = count
        self._time = time
        self._service = service
    
    @property
    def count(self):
        return self._count

    @property
    def time(self):
        return self._time
    
    @property
    def service(self):
        return self._service

    @staticmethod
    def from_response(response):
        throttle_control = response.headers["X-Throttling-Control"]
        throttle_status = [
            substring.split(")")[0] for substring
            in throttle_control.split("(")[1:]
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
        if self._count > threshold or self._time > threshold:
            time.sleep(wait_time)