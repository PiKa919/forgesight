import os
import psutil

def test_rss_measurement_integrity():
    proc = psutil.Process(os.getpid())
    rss_start = proc.memory_info().rss
    assert rss_start > 0
    # Allocate 20MB buffer to verify RSS delta tracking
    data = bytearray(20 * 1024 * 1024)
    rss_after = proc.memory_info().rss
    assert rss_after >= rss_start
    del data
